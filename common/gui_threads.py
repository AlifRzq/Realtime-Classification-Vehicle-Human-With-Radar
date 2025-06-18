# General Library Imports
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import models
from collections import deque
import threading
import time

# PyQt imports
from PySide2.QtCore import QThread, Signal
import pyqtgraph as pg

# Local Imports
from gui_parser import UARTParser
from gui_common import *
from graph_utilities import *

# Logger
import logging
log = logging.getLogger(__name__)

# Classifier Configurables
MAX_NUM_TRACKS = 20 # This could vary depending on the configuration file. Use 20 here as a safe likely maximum to ensure there's enough memory for the classifier

# Expected minimums and maximums to bound the range of colors used for coloring points
SNR_EXPECTED_MIN = 5
SNR_EXPECTED_MAX = 40
SNR_EXPECTED_RANGE = SNR_EXPECTED_MAX - SNR_EXPECTED_MIN

DOPPLER_EXPECTED_MIN = -30
DOPPLER_EXPECTED_MAX = 30
DOPPLER_EXPECTED_RANGE = DOPPLER_EXPECTED_MAX - DOPPLER_EXPECTED_MIN

# Different methods to color the points
COLOR_MODE_SNR = 'SNR'
COLOR_MODE_HEIGHT = 'Height'
COLOR_MODE_DOPPLER = 'Doppler'
COLOR_MODE_TRACK = 'Associated Track'

# Magic Numbers for Target Index TLV
TRACK_INDEX_WEAK_SNR = 253 # Point not associated, SNR too weak
TRACK_INDEX_BOUNDS = 254 # Point not associated, located outside boundary of interest
TRACK_INDEX_NOISE = 255 # Point not associated, considered as noise

# Definisi kelas-kelas keras custom untuk model PointCNN
@tf.keras.utils.register_keras_serializable()
class XConvLayer_PointCNN(tf.keras.layers.Layer):
    """
    Implementasi Layer XConv PointCNN yang lebih mendekati paper asli. 
    """
    def __init__(self, num_neighbors, num_output_features, depth_multiplier=1, l2_reg=0.001, activation='relu', use_bn=True, **kwargs):
        super().__init__(**kwargs)
        self.K = int(num_neighbors)
        self.C_out = int(num_output_features)
        self.D = int(depth_multiplier)
        self.l2_reg = l2_reg
        self.activation_name = activation
        self.activation_fn = tf.keras.layers.Activation(activation)
        self.use_bn = use_bn
        if self.C_out <= 0 or self.D <= 0 or self.C_out % self.D != 0:
            raise ValueError(f"C_out ({self.C_out}) harus positif dan dapat dibagi oleh D ({self.D})")
        self.C_lifted = self.C_out // self.D
        self.mlp_transform = None
        self.feature_lift = None
        self.final_conv = None
        self.bn_lift = None
        self.bn_final = None

    def build(self, input_shape):
        points_xyz_shape, features_shape = input_shape
        self.C_in = features_shape[-1]
        self.N = points_xyz_shape[1]
        if self.N is None or self.C_in is None:
            raise ValueError("Shape input N dan C_in harus diketahui saat build.")
        self.mlp_transform = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, self.K, 3)),
            tf.keras.layers.Conv2D(self.K * self.D, (1, 1), activation=self.activation_name, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)),
            tf.keras.layers.BatchNormalization() if self.use_bn else tf.keras.layers.Lambda(lambda x: x),
            tf.keras.layers.Conv2D(self.K * self.K, (1, self.K), kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        ], name=f'{self.name}_mlp_transform')
        self.feature_lift = tf.keras.layers.Conv2D(self.C_lifted, (1, 1), use_bias=not self.use_bn, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name=f'{self.name}_feature_lift')
        if self.use_bn:
            self.bn_lift = tf.keras.layers.BatchNormalization(name=f'{self.name}_bn_lift')
        self.final_conv = tf.keras.layers.Conv2D(self.C_out, (1, self.K), use_bias=not self.use_bn, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name=f'{self.name}_final_conv')
        if self.use_bn:
            self.bn_final = tf.keras.layers.BatchNormalization(name=f'{self.name}_bn_final')
        
        # --- Panggil build sub-layer secara eksplisit ---
        self.mlp_transform.build(input_shape=(None, 1, self.K, 3))
        self.feature_lift.build(input_shape=(None, None, self.K, self.C_in))
        if self.use_bn:
            self.bn_lift.build(input_shape=(None, None, self.K, self.C_lifted))
        self.final_conv.build(input_shape=(None, None, self.K, self.C_lifted))
        if self.use_bn:
            self.bn_final.build(input_shape=(None, None, 1, self.C_out))
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        points_xyz, features = inputs
        batch_size = tf.shape(points_xyz)[0]
        num_points = tf.shape(points_xyz)[1]
        non_padding_mask = tf.reduce_any(points_xyz != 0.0, axis=-1)
        points_expanded = tf.expand_dims(points_xyz, 2)
        points_tiled = tf.expand_dims(points_xyz, 1)
        pairwise_dist = tf.reduce_sum(tf.square(points_expanded - points_tiled), axis=-1)
        large_distance = tf.constant(1e9, dtype=pairwise_dist.dtype)
        dist_mask = tf.expand_dims(non_padding_mask, 1) & tf.expand_dims(non_padding_mask, 2)
        masked_dist = tf.where(dist_mask, pairwise_dist, large_distance)
        k_neighbors = tf.minimum(self.K, num_points)
        _, indices = tf.math.top_k(-masked_dist, k=k_neighbors, sorted=False)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, num_points, k_neighbors))
        gather_indices = tf.stack([batch_indices, indices], axis=-1)
        neighbor_xyz = tf.gather_nd(points_xyz, gather_indices)
        neighbor_features = tf.gather_nd(features, gather_indices)
        neighbor_padding_mask = tf.gather_nd(non_padding_mask, gather_indices)
        relative_xyz = neighbor_xyz - tf.expand_dims(points_xyz, 2)
        relative_xyz = tf.where(tf.expand_dims(neighbor_padding_mask, -1), relative_xyz, tf.zeros_like(relative_xyz))
        relative_xyz_reshaped = tf.reshape(relative_xyz, [-1, 1, k_neighbors, 3])
        X_weights_reshaped = self.mlp_transform(relative_xyz_reshaped, training=training)
        X_matrix = tf.reshape(X_weights_reshaped, [batch_size, num_points, k_neighbors, k_neighbors])
        lifted_features = self.feature_lift(neighbor_features, training=training)
        if self.use_bn:
            lifted_features = self.bn_lift(lifted_features, training=training)
        lifted_features = self.activation_fn(lifted_features)
        lifted_features = tf.where(tf.expand_dims(neighbor_padding_mask, -1), lifted_features, tf.zeros_like(lifted_features))
        transformed_features = tf.matmul(X_matrix, lifted_features)
        final_features = self.final_conv(transformed_features, training=training)
        if self.use_bn:
            final_features = self.bn_final(final_features, training=training)
        final_features = self.activation_fn(final_features)
        output_features = tf.squeeze(final_features, 2)
        output_features_masked = tf.where(tf.expand_dims(non_padding_mask, -1), output_features, tf.zeros_like(output_features))
        return output_features_masked

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0][0], input_shape[0][1], self.C_out])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_neighbors": self.K,
            "num_output_features": self.C_out,
            "depth_multiplier": self.D,
            "l2_reg": self.l2_reg,
            "activation": self.activation_name,
            "use_bn": self.use_bn
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class PointCNN_Windowed(tf.keras.Model):
    """ Model PointCNN yang diadaptasi untuk input windowed menggunakan SATU layer XConvLayer_PointCNN. """
    def __init__(self, num_classes, window_size, max_points, num_features, **kwargs):
        super(PointCNN_Windowed, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.window_size = window_size
        self.max_points = max_points
        self.num_features = num_features

        # --- Gunakan Hyperparameter PointCNN yang TETAP ---
        k1 = min(self.max_points, 16)  # HP_K1
        d1 = 2  # HP_D1
        f1 = 64  # HP_F1
        fc1_units = 256  # HP_FC1_UNITS
        dropout_rate = 0.35  # HP_DROPOUT
        l2_reg_xconv = 0.001  # L2_REG_XCONV
        l2_reg_fc = 0.01  # L2_REG_FC

        # --- Definisikan sub-model pemroses frame (1 Layer XConv) ---
        frame_input_layer = tf.keras.layers.Input(shape=(self.max_points, self.num_features), name='frame_input')
        points_xyz = tf.keras.layers.Lambda(lambda x: x[:, :, :3], name='extract_xyz')(frame_input_layer)
        features = frame_input_layer
        xconv1_output = XConvLayer_PointCNN(num_neighbors=k1, num_output_features=f1, depth_multiplier=d1, l2_reg=l2_reg_xconv, name='xconv1_pointcnn')([points_xyz, features])
        frame_pooled = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=1) / tf.maximum(tf.reduce_sum(tf.cast(tf.reduce_any(x != 0.0, axis=-1), tf.float32), axis=1, keepdims=True), 1.0),
            name='frame_global_pool'
        )(xconv1_output)
        self.frame_processor = tf.keras.models.Model(inputs=frame_input_layer, outputs=frame_pooled, name='pointcnn_1layer_frame_processor')

        # --- Layer Pemrosesan Temporal ---
        self.time_distributed_processor = tf.keras.layers.TimeDistributed(self.frame_processor, name="td_frame_processor")
        self.temporal_pooling = tf.keras.layers.GlobalAveragePooling1D(name='temporal_average_pooling')
        self.fc1 = tf.keras.layers.Dense(fc1_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg_fc), name='fc1')
        self.bn_fc1 = tf.keras.layers.BatchNormalization(name='bn_fc1')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name='dropout1')
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='output')

    def call(self, inputs, training=None):
        x = self.time_distributed_processor(inputs, training=training)
        x = self.temporal_pooling(x)
        x = self.fc1(x)
        x = self.bn_fc1(x, training=training)
        x = self.dropout1(x, training=training)
        outputs = self.output_layer(x)
        return outputs

    def get_config(self):
        config = super(PointCNN_Windowed, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'window_size': self.window_size,
            'max_points': self.max_points,
            'num_features': self.num_features
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ClassificationThread(QThread):
    result = Signal(dict)
    
    def __init__(self, model_path, scaler_path, window_size=40, max_points=150):
        QThread.__init__(self)
        self.window_size = window_size
        self.max_points = max_points
        self.data_queue = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        # Parameter voting dan filtering yang ditingkatkan dengan adaptive thresholds
        self.recent_predictions = deque(maxlen=7)  # Ditingkatkan dari 5 ke 7 prediksi untuk voting yang lebih stabil
        
        
        self.min_points_thresholds = {0: 6, 1: 10, 2: 8}
        
        self.confidence_thresholds = {0: 0.5, 1: 0.15, 2: 0.95}
        
        # Fallback single thresholds (for backward compatibility)
        self.min_points_threshold = 5  # Default fallback
        self.confidence_threshold = 0.3  # Default fallback
        
        self.no_detection_counter = 0  # Penghitung frame tanpa deteksi
        self.no_detection_limit = 3  # Jumlah frame tanpa deteksi sebelum reset hasil
        
        # Diagnostic counters
        self.diagnostic_stats = {
            'total_frames': 0,
            'below_min_points': 0,
            'classification_attempts': 0,
            'below_confidence': 0,
            'added_to_voting': 0,
            'class_predictions': {0: 0, 1: 0, 2: 0},  # Count per class
            'class_confidences': {0: [], 1: [], 2: []},  # Confidence values per class
            'point_counts': []  # Point count distribution
        }
        
        # Muat model dan scaler
        self.custom_objects = {
            'XConvLayer_PointCNN': XConvLayer_PointCNN, 
            'PointCNN_Windowed': PointCNN_Windowed
        }
        try:
            log.info(f"Memuat model dari {model_path}")
            self.model = models.load_model(model_path, custom_objects=self.custom_objects)
            
            log.info(f"Memuat scaler dari {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Verifikasi scaler
            if hasattr(self.scaler, 'n_features_in_'):
                log.info(f"Scaler berhasil dimuat: {type(self.scaler).__name__} dengan {self.scaler.n_features_in_} fitur")
            else:
                log.warning("Scaler berhasil dimuat tetapi tidak memiliki atribut n_features_in_")
                
            log.info("Model dan scaler berhasil dimuat")
        except Exception as e:
            log.error(f"Error memuat model atau scaler: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.scaler = None
            
        self.class_names = ['Manusia', 'Mobil', 'Motor']
        self.running = True
        log.info("Thread klasifikasi berhasil diinisialisasi dengan voting")
        
    def add_point_cloud(self, point_cloud):
        """Menambahkan point cloud baru ke buffer window"""
        with self.lock:
            self.data_queue.append(point_cloud)
    
    def bootstrap_resample(self, points, n_samples=150):
        """Melakukan resampling bootstrap agar jumlah point = n_samples"""
        if points.shape[0] == 0:
            # Kembalikan array nol dengan jumlah fitur yang benar
            return np.zeros((n_samples, 4))  # Hanya 4 fitur: x, y, z, doppler
        
        # Ambil hanya 4 fitur pertama (x, y, z, doppler)
        # Pastikan dengan eksplisit fitur mana yang diambil
        feature_indices = [0, 1, 2, 3]  # x, y, z, doppler
        
        # Periksa apakah points memiliki cukup kolom
        if points.shape[1] <= 3:
            log.warning(f"Data tidak memiliki cukup fitur: shape={points.shape}")
            # Jika tidak cukup kolom, tambahkan kolom nol untuk doppler
            points_with_doppler = np.zeros((points.shape[0], 4))
            points_with_doppler[:, :points.shape[1]] = points
            points_features = points_with_doppler
        else:
            points_features = points[:, feature_indices]
        
        log.debug(f"Bootstrap resampling: {points.shape} -> {n_samples} points")
        
        if points.shape[0] >= n_samples:
            # Jika jumlah titik cukup atau lebih, ambil n_samples titik secara acak
            indices = np.random.choice(points.shape[0], n_samples, replace=False)
            return points_features[indices]
        else:
            # Jika titik kurang dari n_samples, lakukan bootstrap sampling dengan replacement
            indices = np.random.choice(points.shape[0], n_samples, replace=True)
            return points_features[indices]
    
    def prepare_data(self):
        """Menyiapkan data window untuk prediksi"""
        with self.lock:
            if len(self.data_queue) < self.window_size:
                return None
                
            window_data = list(self.data_queue)
        
        # Periksa apakah terlalu banyak frame kosong dalam window
        empty_frames = sum(1 for frame in window_data if frame.shape[0] < self.min_points_threshold)
        if empty_frames > self.window_size * 0.7:  # Jika >70% frame kosong
            log.debug(f"Terlalu banyak frame kosong dalam window ({empty_frames}/{self.window_size}). Skip prediksi.")
            return None
            
        # Resampling bootstrap untuk setiap frame
        processed_frames = []
        for frame in window_data:
            # Lakukan bootstrap resampling
            resampled = self.bootstrap_resample(frame, self.max_points)
            
            # Terapkan scaling jika scaler tersedia
            if self.scaler is not None:
                try:
                    # Normalisasi data menggunakan scaler
                    resampled = self.scaler.transform(resampled)
                    log.debug("Scaling berhasil diterapkan pada frame")
                except Exception as e:
                    log.error(f"Gagal menerapkan scaling: {e}")
                    
            processed_frames.append(resampled)
            
        # Susun menjadi format yang diharapkan model (1, window_size, max_points, n_features)
        X = np.array([processed_frames], dtype=np.float32)
        return X
    
    def get_adaptive_thresholds(self, pred_class):
        """Dapatkan threshold yang adaptive untuk kelas tertentu"""
        min_points = self.min_points_thresholds.get(pred_class, self.min_points_threshold)
        confidence = self.confidence_thresholds.get(pred_class, self.confidence_threshold)
        return min_points, confidence
    
    def log_diagnostic_info(self):
        """Log diagnostic information setiap 50 frame"""
        if self.diagnostic_stats['total_frames'] % 50 == 0 and self.diagnostic_stats['total_frames'] > 0:
            stats = self.diagnostic_stats
            log.info("=== DIAGNOSTIC REPORT (Last 50 frames) ===")
            log.info(f"Total frames processed: {stats['total_frames']}")
            log.info(f"Below min points: {stats['below_min_points']}")
            log.info(f"Classification attempts: {stats['classification_attempts']}")
            log.info(f"Below confidence threshold: {stats['below_confidence']}")
            log.info(f"Added to voting: {stats['added_to_voting']}")
            
            # Class prediction distribution
            total_preds = sum(stats['class_predictions'].values())
            if total_preds > 0:
                log.info("Class prediction distribution:")
                for class_id, count in stats['class_predictions'].items():
                    percentage = count / total_preds * 100
                    log.info(f"  {self.class_names[class_id]}: {count} ({percentage:.1f}%)")
            
            # Average confidence per class
            log.info("Average confidence per class:")
            for class_id, confidences in stats['class_confidences'].items():
                if confidences:
                    avg_conf = np.mean(confidences)
                    min_conf = np.min(confidences)
                    max_conf = np.max(confidences)
                    log.info(f"  {self.class_names[class_id]}: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")
            
            # Point count distribution
            if stats['point_counts']:
                avg_points = np.mean(stats['point_counts'])
                min_points = np.min(stats['point_counts'])
                max_points = np.max(stats['point_counts'])
                log.info(f"Point count: avg={avg_points:.1f}, min={min_points}, max={max_points}")
            
            log.info("=== END DIAGNOSTIC REPORT ===")
    
    def run(self):
        log.info("Thread klasifikasi mulai berjalan dengan adaptive thresholds")
        log.info(f"Min points thresholds: {self.min_points_thresholds}")
        log.info(f"Confidence thresholds: {self.confidence_thresholds}")
        
        while self.running:
            if self.model is None:
                time.sleep(0.1)
                continue
                
            X = self.prepare_data()
            
            # Update diagnostic counter
            self.diagnostic_stats['total_frames'] += 1
            
            # Cek apakah ada data valid untuk diproses
            if X is not None:
                # Hitung jumlah titik di frame terakhir
                with self.lock:
                    if len(self.data_queue) > 0:
                        last_frame = self.data_queue[-1]
                        points_count = last_frame.shape[0] if last_frame is not None else 0
                    else:
                        points_count = 0
                
                # Update diagnostic stats
                self.diagnostic_stats['point_counts'].append(points_count)
                
                # Gunakan threshold minimum untuk initial check (paling rendah dari semua kelas)
                min_required_points = min(self.min_points_thresholds.values())
                
                # Jika jumlah titik di bawah threshold minimum, skip prediksi
                if points_count < min_required_points:
                    log.debug(f"Jumlah titik ({points_count}) di bawah threshold minimum ({min_required_points}). Skip prediksi.")
                    self.diagnostic_stats['below_min_points'] += 1
                    self.no_detection_counter += 1
                    
                    # Jika counter melebihi limit, reset hasil klasifikasi
                    if self.no_detection_counter >= self.no_detection_limit:
                        log.info(f"Tidak ada deteksi selama {self.no_detection_counter} frame. Reset hasil klasifikasi.")
                        result = {
                            'class': 'Tidak ada objek',
                            'class_id': -1,
                            'confidence': 0.0
                        }
                        self.result.emit(result)
                        self.recent_predictions.clear()  # Reset voting
                    
                    time.sleep(0.05)
                    continue
                
                # Reset counter jika ada titik yang cukup
                self.no_detection_counter = 0
                
                try:
                    # Prediksi klasifikasi
                    log.debug("Melakukan prediksi klasifikasi")
                    self.diagnostic_stats['classification_attempts'] += 1
                    
                    start_time = time.time()
                    pred_probs = self.model.predict(X, verbose=0)
                    inference_time = time.time() - start_time
                    pred_class = np.argmax(pred_probs[0])
                    confidence = pred_probs[0][pred_class]
                    
                    # Update diagnostic stats
                    self.diagnostic_stats['class_predictions'][pred_class] += 1
                    self.diagnostic_stats['class_confidences'][pred_class].append(confidence)
                    
                    # Log semua probabilitas untuk diagnosis
                    probs_str = ", ".join([f"{self.class_names[i]}: {p:.4f}" for i, p in enumerate(pred_probs[0])])
                    log.info(f"Points: {points_count}, Predicted: {self.class_names[pred_class]}, Confidence: {confidence:.4f}")
                    log.debug(f"Probabilitas per kelas: [{probs_str}]")
                    log.info(f"Waktu inferensi: {inference_time:.4f} detik")
                    
                    # RULE-BASED OVERRIDE: Fix bias untuk large objects predicted as motor
                    original_class = pred_class
                    original_confidence = confidence
                    
                    if points_count >= 100 and pred_class == 2:
                        # Banyak points tapi predict motor dengan confidence tidak sangat tinggi
                        log.warning(f"🚗 OVERRIDE: {points_count} points, forcing CAR instead of MOTOR")
                        log.warning(f"   Original: Motor (conf: {confidence:.3f}) → Override: Car (conf: 0.7)")
                        pred_class = 1  # Force car
                        confidence = 0.85  # Set reasonable confidence
                    
                    # Log threshold comparison for all classes
                    for class_id in range(len(self.class_names)):
                        class_min_points = self.min_points_thresholds.get(class_id, self.min_points_threshold)
                        class_confidence = self.confidence_thresholds.get(class_id, self.confidence_threshold)
                        prob_for_class = pred_probs[0][class_id]
                        points_ok = points_count >= class_min_points
                        conf_ok = prob_for_class >= class_confidence
                        status = "✓" if (points_ok and conf_ok) else "✗"
                        log.info(f"{status} {self.class_names[class_id]}: prob={prob_for_class:.4f} (>={class_confidence:.3f}), points={points_count} (>={class_min_points})")
                    
                    # Gunakan adaptive thresholds berdasarkan predicted class
                    class_min_points, class_confidence_threshold = self.get_adaptive_thresholds(pred_class)
                    
                    # Check class-specific point threshold
                    if points_count < class_min_points:
                        log.debug(f"Points ({points_count}) di bawah threshold untuk {self.class_names[pred_class]} ({class_min_points}). Skip.")
                        self.diagnostic_stats['below_min_points'] += 1
                        time.sleep(0.05)
                        continue
                    
                    # Check class-specific confidence threshold
                    if confidence >= class_confidence_threshold:
                        # Tambahkan prediksi baru ke daftar untuk voting
                        self.recent_predictions.append(pred_class)
                        self.diagnostic_stats['added_to_voting'] += 1
                        log.info(f"Prediksi ditambahkan ke voting: {self.class_names[pred_class]} (conf: {confidence:.4f} >= {class_confidence_threshold:.4f})")
                        
                        # SPECIAL DEBUG FOR MOTOR ADDED TO VOTING
                        if pred_class == 2:
                            log.warning(f"🏍️ MOTOR ADDED TO VOTING! Current voting queue: {[self.class_names[p] for p in self.recent_predictions]}")
                    else:
                        log.info(f"Confidence ({confidence:.4f}) di bawah threshold {self.class_names[pred_class]} ({class_confidence_threshold:.4f}). Skip untuk voting.")
                        self.diagnostic_stats['below_confidence'] += 1
                        
                        # SPECIAL DEBUG FOR MOTOR REJECTED
                        if pred_class == 2:
                            log.warning(f"🏍️ MOTOR REJECTED! Confidence {confidence:.4f} < threshold {class_confidence_threshold:.4f}")
                    
                    # Lakukan voting jika ada cukup data
                    if len(self.recent_predictions) > 0:
                        # Voting berdasarkan mayoritas
                        from collections import Counter
                        vote_counts = Counter(self.recent_predictions)
                        vote_result, vote_count = vote_counts.most_common(1)[0]
                        
                        # Hitung persentase suara untuk kelas terpilih
                        vote_percentage = vote_count / len(self.recent_predictions)
                        
                        log.info(f"Hasil voting: {self.class_names[vote_result]} ({vote_percentage:.2%})")
                        
                        # SPECIAL DEBUG FOR VOTING RESULTS
                        vote_breakdown = ", ".join([f"{self.class_names[k]}: {v}" for k, v in vote_counts.items()])
                        log.info(f"Voting breakdown: {vote_breakdown}")
                        
                        # Gunakan hasil voting langsung tanpa bias mechanism
                        result = {
                            'class': self.class_names[vote_result],
                            'class_id': vote_result,
                            'confidence': float(confidence),  # Tetap gunakan confidence prediksi terbaru
                            'vote_percentage': float(vote_percentage)
                        }
                    else:
                        # Jika belum ada data untuk voting, gunakan "Tidak ada objek"
                        result = {
                            'class': 'Tidak ada objek',
                            'class_id': -1,
                            'confidence': 0.0
                        }
                    
                    log.info(f"Hasil klasifikasi: {result['class']} dengan confidence {result.get('confidence', 0.0):.4f}")
                    
                    # SPECIAL DEBUG FOR MOTOR UI UPDATE
                    if result.get('class_id') == 2:
                        log.warning(f"🏍️ EMITTING MOTOR RESULT TO UI: {result}")
                    
                    self.result.emit(result)
                    
                except Exception as e:
                    log.error(f"Error saat prediksi: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Jika tidak ada data yang cukup dalam window, tambah counter
                self.no_detection_counter += 1
                if self.no_detection_counter >= self.no_detection_limit:
                    log.debug(f"Tidak ada window data yang valid selama {self.no_detection_counter} iterasi.")
            
            # Log diagnostic info setiap 50 frame
            self.log_diagnostic_info()
            
            # Sleep untuk tidak membebankan CPU terlalu berat
            time.sleep(0.05)
        
        log.info("Thread klasifikasi berhenti")
    
    def stop(self):
        log.info("Menghentikan thread klasifikasi")
        self.running = False
        self.terminate()

class parseUartThread(QThread):
    fin = Signal(dict)

    def __init__(self, uParser):
        QThread.__init__(self)
        self.parser = uParser

    def run(self):
        if(self.parser.parserType == "SingleCOMPort"):
            outputDict = self.parser.readAndParseUartSingleCOMPort()
        else:
            outputDict = self.parser.readAndParseUartDoubleCOMPort()

        self.fin.emit(outputDict)

    def stop(self):
        self.terminate()

class sendCommandThread(QThread):
    done = Signal()

    def __init__(self, uParser, command):
        QThread.__init__(self)
        self.parser = uParser
        self.command = command

    def run(self):
        self.parser.sendLine(self.command)
        self.done.emit()

class updateQTTargetThread3D(QThread):
    done = Signal()

    def __init__(self, pointCloud, targets, scatter, pcplot, numTargets, ellipsoids, coords, 
                 colorGradient=None, classifierOut=[], zRange=[-3, 3], pointColorMode="", 
                 drawTracks=True, trackColorMap=None, pointBounds={'enabled': False},
                 classificationResult=None):  # Tambahkan parameter hasil klasifikasi

        QThread.__init__(self)
        self.pointCloud = pointCloud
        self.targets = targets
        self.scatter = scatter
        self.pcplot = pcplot
        self.colorArray = ('r','g','b','w')
        self.numTargets = numTargets
        self.ellipsoids = ellipsoids
        self.coordStr = coords
        self.classifierOut = classifierOut
        self.zRange = zRange
        self.colorGradient = colorGradient
        self.pointColorMode = pointColorMode
        self.drawTracks = drawTracks
        self.trackColorMap = trackColorMap
        self.pointBounds = pointBounds
        self.classificationResult = classificationResult  # Simpan hasil klasifikasi

        # This ignores divide by 0 errors when calculating the log2
        np.seterr(divide = 'ignore')

    def drawTrack(self, track, trackColor):
        # Get necessary track data
        tid = int(track[0])
        x = track[1]
        y = track[2]
        z = track[3]
        track = self.ellipsoids[tid]
        mesh = getBoxLinesCoords(x,y,z)
        track.setData(pos=mesh,color=trackColor,width=2,antialias=True,mode='lines')
        track.setVisible(True)

    # Return transparent color if pointBounds is enabled and point is outside pointBounds
    # Otherwise, color the point depending on which color mode we are in
    def getPointColors(self, i):
        if (self.pointBounds['enabled']):
            xyz_coords = self.pointCloud[i,0:3]
            if (xyz_coords[0] < self.pointBounds['minX']
                or xyz_coords[0] > self.pointBounds['maxX']
                or xyz_coords[1] < self.pointBounds['minY']
                or xyz_coords[1] > self.pointBounds['maxY']
                or xyz_coords[2] < self.pointBounds['minZ']
                or xyz_coords[2] > self.pointBounds['maxZ']):
                return pg.glColor((0,0,0,0))

        # Color the points by their SNR
        if (self.pointColorMode == COLOR_MODE_SNR):
            snr = self.pointCloud[i,4]
            # SNR value is out of expected bounds, make it white
            if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((snr-SNR_EXPECTED_MIN)/SNR_EXPECTED_RANGE))

        # Color the points by their Height
        elif (self.pointColorMode == COLOR_MODE_HEIGHT):
            zs = self.pointCloud[i, 2]
            # Points outside expected z range, make it white
            if (zs < self.zRange[0]) or (zs > self.zRange[1]):
                return pg.glColor('w')
            else:
                colorRange = self.zRange[1]+abs(self.zRange[0])
                zs = self.zRange[1] - zs
                return pg.glColor(self.colorGradient.getColor(abs(zs/colorRange)))

        # Color Points by their doppler
        elif(self.pointColorMode == COLOR_MODE_DOPPLER):
            doppler = self.pointCloud[i,3]
            # Doppler value is out of expected bounds, make it white
            if (doppler < DOPPLER_EXPECTED_MIN) or (doppler > DOPPLER_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((doppler-DOPPLER_EXPECTED_MIN)/DOPPLER_EXPECTED_RANGE))

        # Color the points by their associate track
        elif (self.pointColorMode == COLOR_MODE_TRACK):
            trackIndex = int(self.pointCloud[i, 6])
            # trackIndex of 253, 254, or 255 indicates a point isn't associated to a track, so check for those magic numbers here
            if (trackIndex == TRACK_INDEX_WEAK_SNR or trackIndex == TRACK_INDEX_BOUNDS or trackIndex == TRACK_INDEX_NOISE):
                return pg.glColor('w')
            else:
                # Catch any errors that may occur if track or point index go out of bounds
                try:
                    return self.trackColorMap[trackIndex]
                except Exception as e:
                    log.error(e)
                    return pg.glColor('w')

        # Unknown Color Option, make all points green
        else:
            return pg.glColor('g')

    def run(self):
        # Clear all previous targets
        for e in self.ellipsoids:
            if (e.visible()):
                e.hide()

        try:
            # Create a list of just X, Y, Z values to be plotted
            if(self.pointCloud is not None):
                toPlot = self.pointCloud[:, 0:3]
                # Determine the size of each point based on its SNR
                with np.errstate(divide='ignore'):
                    size = np.log2(self.pointCloud[:, 4])
                # Each color is an array of 4 values, so we need an numPoints*4 size 2d array to hold these values
                pointColors = np.zeros((self.pointCloud.shape[0], 4))
                # Set the color of each point
                for i in range(self.pointCloud.shape[0]):
                    pointColors[i] = self.getPointColors(i)
                # Plot the points
                self.scatter.setData(pos=toPlot, color=pointColors, size=size)
                # Make the points visible
                self.scatter.setVisible(True)
            else:
                # Make the points invisible if none are detected.
                self.scatter.setVisible(False)
        except Exception as e:
            log.error(f"Unable to draw point cloud: {e}")
            log.error("Ignoring and continuing execution...")

        # Graph the targets
        try:
            if (self.drawTracks):
                if (self.targets is not None):
                    for track in self.targets:
                        trackID = int(track[0])
                        trackColor = self.trackColorMap[trackID]
                        self.drawTrack(track,trackColor)
        except Exception as e:
            log.error(f"Unable to draw all tracks: {e}")
            log.error("Ignoring and continuing execution...")
            
        # Tampilkan hasil klasifikasi jika ada
        try:
            if self.classificationResult is not None:
                class_name = self.classificationResult.get('class', 'Tidak diketahui')
                confidence = self.classificationResult.get('confidence', 0.0)
                
                # Tambahkan teks klasifikasi ke plot
                class_text = pg.TextItem(
                    text=f"Kelas: {class_name}\nConfidence: {confidence:.2f}",
                    color=(255, 255, 255),
                    anchor=(0, 0)
                )
                self.pcplot.addItem(class_text)
                log.debug(f"Menampilkan hasil klasifikasi di plot: {class_name} ({confidence:.2f})")
        except Exception as e:
            log.error(f"Gagal menampilkan hasil klasifikasi: {e}")

        self.done.emit()

    def stop(self):
        self.terminate()
