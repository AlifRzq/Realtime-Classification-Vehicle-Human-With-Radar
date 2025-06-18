# -*- coding: utf-8 -*-
"""
Final PointCNN Training Script with Windowing (1 XConv Layer, Fixed Hyperparameters)
(Applies PointCNN blocks per frame using TimeDistributed and temporal pooling)
*** MODIFIED FOR MULTI-GPU TRAINING WITH MirroredStrategy ***
*** USES PointCNN formulation closer to original paper (v2 - Reshape Fix) ***
*** Includes ModelCheckpoint and StandardScaler saving ***
"""

# --- Standard Libraries ---
import os
import time
import gc # Optional memory management

# --- Data Handling ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # <<< Import joblib untuk menyimpan scaler

# --- TensorFlow & Keras ---
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D # Implicitly needed

# --- Metrics ---
from sklearn.metrics import confusion_matrix, classification_report


# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Data Paths ---
# !! PENTING: Sesuaikan path ini !!
DATA_PATH_PREFIX = '/kaggle/input/data-klasifikasi/' # Contoh path Kaggle

# --- Windowing Parameters ---
WINDOW_SIZE = 40        # Jumlah frame (timestamp) per sekuen window
STEP = 1                # Langkah untuk sliding window

# --- Model & Preprocessing Hyperparameters ---
MAX_POINTS = 150        # Jumlah titik maksimum per frame setelah padding/truncating
# --- Hyperparameter PointCNN Layer 1 (TETAP) ---
HP_K1 = 16             # K: Jumlah tetangga untuk layer XConv pertama
HP_D1 = 2              # D: Depth multiplier untuk layer XConv pertama
HP_F1 = 64             # C_out: Jumlah fitur output untuk layer XConv pertama
# --- Hyperparameter FC Layer (TETAP) ---
HP_FC1_UNITS = 256     # Jumlah unit di Dense layer setelah pooling temporal
HP_DROPOUT = 0.35      # Rate Dropout
L2_REG_XCONV = 0.001   # Regularisasi L2 untuk MLP internal XConv
L2_REG_FC = 0.01       # Regularisasi L2 untuk Dense layer akhir
# --------------------------------------------------------------------------

# --- Training Hyperparameters ---
LEARNING_RATE = 5e-4    # Learning rate awal
WEIGHT_DECAY = 0.004   # Weight decay untuk AdamW
# !! BATCH_SIZE sekarang adalah ukuran batch PER REPLIKA !!
PER_REPLICA_BATCH_SIZE = 16 # Sesuaikan sesuai VRAM GPU
EPOCHS = 50            # Jumlah epoch maksimum (dapat dihentikan oleh EarlyStopping)
EARLY_STOPPING_PATIENCE = 10 # Kesabaran untuk EarlyStopping (epoch tanpa peningkatan val_loss)
REDUCE_LR_PATIENCE = 7      # Kesabaran untuk ReduceLROnPlateau

# --- Identifikasi Run & Path Output ---
RUN_TIMESTAMP = int(time.time())
# Nama run disesuaikan untuk mencerminkan hyperparameter tetap dan 1 layer
RUN_NAME = f"pointcnn_1L_fixed_K1_{HP_K1}_D1_{HP_D1}_F1_{HP_F1}_FC_{HP_FC1_UNITS}_{RUN_TIMESTAMP}"
OUTPUT_BASE_DIR = f'./results_{RUN_NAME}'
LOG_DIR = os.path.join(OUTPUT_BASE_DIR, 'logs')
GRAPHICS_DIR = os.path.join(OUTPUT_BASE_DIR, 'graphics')
# Path penyimpanan model terbaik
MODEL_BEST_LOSS_PATH = os.path.join(OUTPUT_BASE_DIR, f'best_model_{RUN_NAME}_loss.keras')
MODEL_BEST_ACC_PATH = os.path.join(OUTPUT_BASE_DIR, f'best_model_{RUN_NAME}_acc.keras')
# <<< Path penyimpanan scaler >>>
SCALER_PATH = os.path.join(OUTPUT_BASE_DIR, f'scaler_{RUN_NAME}.joblib')

# --- Definisi Kelas ---
CLASS_NAMES = ['Manusia', 'Mobil', 'Motor']
Figure_Name = ['Human', 'Car', 'Motorcycle'] # Nama untuk plot
NUM_CLASSES = len(CLASS_NAMES)

# ==============================================================================
# --- Setup & System Checks ---
# ==============================================================================

# --- Setup Direktori Output ---
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(GRAPHICS_DIR, exist_ok=True)
print(f"Output akan disimpan di: {OUTPUT_BASE_DIR}")

# --- Cek GPU & Setup Strategy ---
print("--- Informasi Sistem & Setup Strategy ---")
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Ditemukan {len(gpus)} GPU(s): {gpus}")
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth diaktifkan untuk GPU(s).")
        strategy = tf.distribute.MirroredStrategy()
        print(f"Menggunakan MirroredStrategy dengan {strategy.num_replicas_in_sync} replika.")
    except RuntimeError as e:
        print(f"Error saat mengatur memory growth atau membuat strategy: {e}")
        print("Kembali ke strategy default.")
        strategy = tf.distribute.get_strategy()
else:
    print("Tidak ada GPU terdeteksi. Berjalan di CPU menggunakan strategy default.")
    strategy = tf.distribute.get_strategy()

GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync
print(f"Per-Replica Batch Size: {PER_REPLICA_BATCH_SIZE}")
print(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")
print("-" * 30)


# ==============================================================================
# --- Fungsi Loading dan Preprocessing Data ---
# ==============================================================================
# (Fungsi load_and_combine_data, group_point_clouds_by_timestamp,
#  pad_all_frames, create_windowed_data_from_padded, augment_windowed_data tetap sama)
def load_and_combine_data(path_prefix, class_names):
    all_data = []; print("Memuat data...")
    for i, class_name in enumerate(class_names):
        found = False;
        for case_variation in [class_name.lower(), class_name.capitalize(), class_name.upper()]:
             filename_variation = f"dataset{case_variation}.csv"; file_path = os.path.join(path_prefix, filename_variation)
             if os.path.exists(file_path):
                  try: df = pd.read_csv(file_path); df['label'] = i; all_data.append(df); print(f"  Loaded {filename_variation} ({len(df)} baris)"); found = True; break
                  except Exception as e: print(f"ERROR: Gagal memuat {file_path}: {e}"); return None
        if not found: print(f"ERROR: File CSV tidak ditemukan untuk kelas '{class_name}' di {path_prefix}"); return None
    if not all_data: print("ERROR: Tidak ada file data yang dimuat."); return None
    combined = pd.concat(all_data, ignore_index=True); print(f"Data digabungkan. Total baris: {len(combined)}"); print("Distribusi Label:\n", combined['label'].value_counts().sort_index()); return combined

def scale_features(df, features_list, scaler_save_path=None): # <<< Tambahkan argumen path scaler
    """Menskalakan fitur yang ditentukan menggunakan StandardScaler dan menyimpan scaler."""
    print("Menskalakan fitur...");
    if not all(col in df.columns for col in features_list): missing = [col for col in features_list if col not in df.columns]; print(f"ERROR: Fitur hilang: {missing}"); return None, None
    scaler = StandardScaler(); df[features_list] = scaler.fit_transform(df[features_list].values)
    print("Fitur diskalakan.")
    # <<< Simpan scaler jika path diberikan >>>
    if scaler_save_path:
        try:
            joblib.dump(scaler, scaler_save_path)
            print(f"Scaler disimpan ke: {scaler_save_path}")
        except Exception as e:
            print(f"ERROR: Gagal menyimpan scaler ke {scaler_save_path}: {e}")
            # Lanjutkan meskipun gagal menyimpan, tapi beri peringatan
    # <<< Akhir penambahan penyimpanan scaler >>>
    return df, scaler # Kembalikan scaler juga

def group_point_clouds_by_timestamp(df, feature_cols, label_col='label', time_col='timestamp'):
    print("Mengelompokkan point clouds berdasarkan timestamp...")
    if time_col not in df.columns: print(f"ERROR: Kolom timestamp '{time_col}' tidak ditemukan."); return None, None, None
    df = df.sort_values(by=time_col); grouped = df.groupby(time_col)
    point_clouds = []; labels_list = []; timestamps_list = []
    for timestamp, group in grouped: point_clouds.append(group[feature_cols].values.astype(np.float32)); labels_list.append(group[label_col].iloc[0]); timestamps_list.append(timestamp)
    print(f"Selesai mengelompokkan. Ditemukan {len(point_clouds)} timestamps unik (frame).")
    valid_indices = [i for i, pc in enumerate(point_clouds) if pc.shape[0] > 0]
    if len(valid_indices) < len(point_clouds): print(f"Peringatan: Ditemukan {len(point_clouds) - len(valid_indices)} frame kosong setelah grouping. Menghapusnya."); point_clouds = [point_clouds[i] for i in valid_indices]; labels_list = [labels_list[i] for i in valid_indices]; timestamps_list = [timestamps_list[i] for i in valid_indices]
    return np.array(point_clouds, dtype=object), np.array(labels_list), np.array(timestamps_list)

def pad_all_frames(point_clouds_list, max_points):
    print(f"Padding semua frame ke {max_points} titik...")
    if not point_clouds_list.size: return np.array([], dtype=np.float32)
    num_features = 4
    for cloud in point_clouds_list:
        if isinstance(cloud, np.ndarray) and cloud.ndim == 2 and cloud.shape[1] > 0: num_features = cloud.shape[1]; break
    print(f"  Terdeteksi/Menggunakan {num_features} fitur per titik.")
    padded_data = np.zeros((len(point_clouds_list), max_points, num_features), dtype=np.float32)
    for i, cloud in enumerate(point_clouds_list):
        if isinstance(cloud, np.ndarray) and cloud.ndim == 2 and cloud.shape[0] > 0:
             num_points_to_copy = min(cloud.shape[0], max_points); current_features = cloud.shape[1]
             if current_features == num_features: padded_data[i, :num_points_to_copy] = cloud[:num_points_to_copy]
             elif current_features < num_features: padded_data[i, :num_points_to_copy, :current_features] = cloud[:num_points_to_copy]
             else: padded_data[i, :num_points_to_copy] = cloud[:num_points_to_copy, :num_features]
    print(f"Padding selesai. Shape data padded: {padded_data.shape}")
    return padded_data

def create_windowed_data_from_padded(padded_data, labels, timestamps, window_size, step):
    print(f"Membuat window ukuran {window_size} dengan langkah {step}...")
    num_samples = padded_data.shape[0]
    if num_samples < window_size: print(f"ERROR: Tidak cukup frame ({num_samples}) untuk window ukuran {window_size}."); return None, None
    windowed_data = []; windowed_labels = []
    for i in range(0, num_samples - window_size + 1, step): window = padded_data[i : i + window_size]; label = labels[i + window_size - 1]; windowed_data.append(window); windowed_labels.append(label)
    if not windowed_data: print("ERROR: Tidak ada window yang dibuat."); return None, None
    print(f"Dibuat {len(windowed_data)} window.")
    return np.array(windowed_data, dtype=np.float32), np.array(windowed_labels)

# --- Fungsi Augmentasi Data ---
def rotate_point_cloud(point_cloud):
    theta = np.random.uniform(0, 2 * np.pi); cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t, 0], [sin_t,  cos_t, 0], [0, 0, 1]], dtype=np.float32)
    augmented_cloud = np.copy(point_cloud); non_padding_mask = np.any(point_cloud != 0, axis=1)
    if np.any(non_padding_mask): coords = point_cloud[non_padding_mask, :3]; rotated_coords = coords @ rotation_matrix; augmented_cloud[non_padding_mask, :3] = rotated_coords
    return augmented_cloud
def jitter_point_cloud(point_cloud, sigma=0.01):
    jittered_cloud = np.copy(point_cloud); non_padding_mask = np.any(point_cloud != 0, axis=1)
    if np.any(non_padding_mask): noise = np.random.normal(0, sigma, size=jittered_cloud[non_padding_mask, :3].shape).astype(np.float32); jittered_cloud[non_padding_mask, :3] += noise
    return jittered_cloud
def scale_point_cloud(point_cloud, scale_low=0.8, scale_high=1.2):
    scale = np.float32(np.random.uniform(scale_low, scale_high)); scaled_cloud = np.copy(point_cloud); non_padding_mask = np.any(point_cloud != 0, axis=1)
    if np.any(non_padding_mask): scaled_cloud[non_padding_mask, :3] *= scale
    return scaled_cloud
def augment_frame(point_cloud):
    augment_choice = np.random.choice(['rotate', 'jitter', 'scale', 'none'])
    if augment_choice == 'rotate': return rotate_point_cloud(point_cloud)
    elif augment_choice == 'jitter': return jitter_point_cloud(point_cloud)
    elif augment_choice == 'scale': return scale_point_cloud(point_cloud)
    else: return point_cloud
def augment_windowed_data(X_data, y_data, apply_prob=0.75):
    X_augmented = []; y_augmented = []
    num_windows = X_data.shape[0]; print(f"Augmentasi {num_windows} window training...")
    for window_idx in range(num_windows):
        original_window = X_data[window_idx]; original_label = y_data[window_idx]
        X_augmented.append(original_window); y_augmented.append(original_label)
        augmented_window = np.copy(original_window)
        for frame_idx in range(original_window.shape[0]):
            if np.random.rand() < apply_prob: augmented_window[frame_idx] = augment_frame(original_window[frame_idx])
        X_augmented.append(augmented_window); y_augmented.append(original_label)
        if (window_idx + 1) % (max(1, num_windows // 20)) == 0: print(f"  Progres Augmentasi: {((window_idx + 1) / num_windows) * 100:.1f}%", end='\r')
    print("\nAugmentasi selesai. Mengacak data hasil augmentasi...")
    X_final = np.array(X_augmented, dtype=np.float32); y_final = np.array(y_augmented)
    indices = np.arange(len(X_final)); np.random.shuffle(indices)
    print(f"Total sampel training setelah augmentasi: {len(X_final)}")
    return X_final[indices], y_final[indices]


# ==============================================================================
# --- Definisi Model (XConvLayer_PointCNN & PointCNN_Windowed dengan 1 Layer) ---
# ==============================================================================
print("--- Mendefinisikan Komponen Model (Formulasi PointCNN - 1 Layer) ---")

@tf.keras.utils.register_keras_serializable()
class XConvLayer_PointCNN(layers.Layer):
    """
    Implementasi Layer XConv PointCNN yang lebih mendekati paper asli. (v2 - Reshape Fix)
    """
    def __init__(self, num_neighbors, num_output_features, depth_multiplier=1, l2_reg=0.001, activation='relu', use_bn=True, **kwargs):
        super().__init__(**kwargs)
        self.K = int(num_neighbors); self.C_out = int(num_output_features); self.D = int(depth_multiplier)
        self.l2_reg = l2_reg; self.activation_name = activation; self.activation_fn = layers.Activation(activation); self.use_bn = use_bn
        if self.C_out <= 0 or self.D <= 0 or self.C_out % self.D != 0: raise ValueError(f"C_out ({self.C_out}) harus positif dan dapat dibagi oleh D ({self.D})")
        self.C_lifted = self.C_out // self.D
        self.mlp_transform = None; self.feature_lift = None; self.final_conv = None; self.bn_lift = None; self.bn_final = None

    def build(self, input_shape):
        points_xyz_shape, features_shape = input_shape; self.C_in = features_shape[-1]; self.N = points_xyz_shape[1]
        if self.N is None or self.C_in is None: raise ValueError("Shape input N dan C_in harus diketahui saat build.")
        self.mlp_transform = tf.keras.Sequential([ layers.InputLayer(input_shape=(1, self.K, 3)), layers.Conv2D(self.K * self.D, (1, 1), activation=self.activation_name, kernel_regularizer=regularizers.l2(self.l2_reg)), layers.BatchNormalization() if self.use_bn else layers.Lambda(lambda x: x), layers.Conv2D(self.K * self.K, (1, self.K), kernel_regularizer=regularizers.l2(self.l2_reg)) ], name=f'{self.name}_mlp_transform')
        self.feature_lift = layers.Conv2D(self.C_lifted, (1, 1), use_bias=not self.use_bn, kernel_regularizer=regularizers.l2(self.l2_reg), name=f'{self.name}_feature_lift')
        if self.use_bn: self.bn_lift = layers.BatchNormalization(name=f'{self.name}_bn_lift')
        self.final_conv = layers.Conv2D(self.C_out, (1, self.K), use_bias=not self.use_bn, kernel_regularizer=regularizers.l2(self.l2_reg), name=f'{self.name}_final_conv')
        if self.use_bn: self.bn_final = layers.BatchNormalization(name=f'{self.name}_bn_final')
        # --- Panggil build sub-layer secara eksplisit ---
        self.mlp_transform.build(input_shape=(None, 1, self.K, 3))
        self.feature_lift.build(input_shape=(None, None, self.K, self.C_in))
        if self.use_bn: self.bn_lift.build(input_shape=(None, None, self.K, self.C_lifted))
        self.final_conv.build(input_shape=(None, None, self.K, self.C_lifted))
        if self.use_bn: self.bn_final.build(input_shape=(None, None, 1, self.C_out))
        # ---------------------------------------------
        super().build(input_shape) # Panggil build parent di akhir

    def call(self, inputs, training=None):
        points_xyz, features = inputs; batch_size = tf.shape(points_xyz)[0]; num_points = tf.shape(points_xyz)[1]
        non_padding_mask = tf.reduce_any(points_xyz != 0.0, axis=-1); points_expanded = tf.expand_dims(points_xyz, 2); points_tiled = tf.expand_dims(points_xyz, 1)
        pairwise_dist = tf.reduce_sum(tf.square(points_expanded - points_tiled), axis=-1); large_distance = tf.constant(1e9, dtype=pairwise_dist.dtype)
        dist_mask = tf.expand_dims(non_padding_mask, 1) & tf.expand_dims(non_padding_mask, 2); masked_dist = tf.where(dist_mask, pairwise_dist, large_distance)
        k_neighbors = tf.minimum(self.K, num_points); _, indices = tf.math.top_k(-masked_dist, k=k_neighbors, sorted=False)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, num_points, k_neighbors)); gather_indices = tf.stack([batch_indices, indices], axis=-1)
        neighbor_xyz = tf.gather_nd(points_xyz, gather_indices); neighbor_features = tf.gather_nd(features, gather_indices); neighbor_padding_mask = tf.gather_nd(non_padding_mask, gather_indices)
        relative_xyz = neighbor_xyz - tf.expand_dims(points_xyz, 2); relative_xyz = tf.where(tf.expand_dims(neighbor_padding_mask, -1), relative_xyz, tf.zeros_like(relative_xyz))
        relative_xyz_reshaped = tf.reshape(relative_xyz, [-1, 1, k_neighbors, 3]); X_weights_reshaped = self.mlp_transform(relative_xyz_reshaped, training=training)
        X_matrix = tf.reshape(X_weights_reshaped, [batch_size, num_points, k_neighbors, k_neighbors]) # FIX: Use k_neighbors
        lifted_features = self.feature_lift(neighbor_features, training=training)
        if self.use_bn: lifted_features = self.bn_lift(lifted_features, training=training)
        lifted_features = self.activation_fn(lifted_features); lifted_features = tf.where(tf.expand_dims(neighbor_padding_mask, -1), lifted_features, tf.zeros_like(lifted_features))
        transformed_features = tf.matmul(X_matrix, lifted_features); final_features = self.final_conv(transformed_features, training=training)
        if self.use_bn: final_features = self.bn_final(final_features, training=training)
        final_features = self.activation_fn(final_features); output_features = tf.squeeze(final_features, 2)
        output_features_masked = tf.where(tf.expand_dims(non_padding_mask, -1), output_features, tf.zeros_like(output_features))
        return output_features_masked

    def compute_output_shape(self, input_shape): return tf.TensorShape([input_shape[0][0], input_shape[0][1], self.C_out])
    def get_config(self): config = super().get_config(); config.update({ "num_neighbors": self.K, "num_output_features": self.C_out, "depth_multiplier": self.D, "l2_reg": self.l2_reg, "activation": self.activation_name, "use_bn": self.use_bn }); return config
    @classmethod
    def from_config(cls, config): return cls(**config)

@tf.keras.utils.register_keras_serializable()
class PointCNN_Windowed(tf.keras.Model):
    """ Model PointCNN yang diadaptasi untuk input windowed menggunakan SATU layer XConvLayer_PointCNN. """
    def __init__(self, num_classes, window_size, max_points, num_features, **kwargs):
        super(PointCNN_Windowed, self).__init__(**kwargs)
        self.num_classes = num_classes; self.window_size = window_size
        self.max_points = max_points; self.num_features = num_features

        # --- Gunakan Hyperparameter PointCNN yang TETAP ---
        k1 = min(self.max_points, HP_K1); d1 = HP_D1; f1 = HP_F1
        fc1_units = HP_FC1_UNITS; dropout_rate = HP_DROPOUT
        l2_reg_xconv = L2_REG_XCONV; l2_reg_fc = L2_REG_FC

        print(f"Parameter PointCNN_Windowed (1 Layer):")
        print(f"  Layer 1: K1={k1}, D1={d1}, F1={f1}")
        print(f"  FC: Units={fc1_units}, Dropout={dropout_rate}")

        # --- Definisikan sub-model pemroses frame (1 Layer XConv) ---
        frame_input_layer = layers.Input(shape=(self.max_points, self.num_features), name='frame_input')
        points_xyz = layers.Lambda(lambda x: x[:, :, :3], name='extract_xyz')(frame_input_layer)
        features = frame_input_layer
        xconv1_output = XConvLayer_PointCNN(num_neighbors=k1, num_output_features=f1, depth_multiplier=d1, l2_reg=l2_reg_xconv, name='xconv1_pointcnn')([points_xyz, features])
        frame_pooled = layers.Lambda( lambda x: tf.reduce_sum(x, axis=1) / tf.maximum(tf.reduce_sum(tf.cast(tf.reduce_any(x != 0.0, axis=-1), tf.float32), axis=1, keepdims=True), 1.0), name='frame_global_pool')(xconv1_output)
        self.frame_processor = models.Model(inputs=frame_input_layer, outputs=frame_pooled, name='pointcnn_1layer_frame_processor')

        # --- Layer Pemrosesan Temporal ---
        self.time_distributed_processor = layers.TimeDistributed(self.frame_processor, name="td_frame_processor")
        self.temporal_pooling = layers.GlobalAveragePooling1D(name='temporal_average_pooling')
        self.fc1 = layers.Dense(fc1_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg_fc), name='fc1')
        self.bn_fc1 = layers.BatchNormalization(name='bn_fc1')
        self.dropout1 = layers.Dropout(dropout_rate, name='dropout1')
        self.output_layer = layers.Dense(self.num_classes, activation='softmax', name='output')

    def call(self, inputs, training=None):
        x = self.time_distributed_processor(inputs, training=training)
        x = self.temporal_pooling(x)
        x = self.fc1(x)
        x = self.bn_fc1(x, training=training)
        x = self.dropout1(x, training=training)
        outputs = self.output_layer(x)
        return outputs

    def get_config(self): config = super(PointCNN_Windowed, self).get_config(); config.update({ 'num_classes': self.num_classes, 'window_size': self.window_size, 'max_points': self.max_points, 'num_features': self.num_features }); return config
    @classmethod
    def from_config(cls, config): return cls(**config)

print("Definisi model 1 Layer XConv selesai (Menggunakan XConvLayer_PointCNN).")
print("-" * 30)


# ==============================================================================
# --- Eksekusi Utama ---
# ==============================================================================

# 1. Muat Data
combined_data_loaded = load_and_combine_data(DATA_PATH_PREFIX, CLASS_NAMES)
if combined_data_loaded is None: exit()

# 2. Skalakan Fitur <<< MODIFIKASI DI SINI >>>
features_to_scale = ['x', 'y', 'z', 'doppler']
# Teruskan SCALER_PATH ke fungsi scale_features
scaled_data, fitted_scaler = scale_features(combined_data_loaded, features_to_scale, scaler_save_path=SCALER_PATH)
if scaled_data is None or fitted_scaler is None: print("ERROR: Gagal menskalakan fitur atau menyimpan scaler."); exit()
del combined_data_loaded; gc.collect()

# 3. Kelompokkan berdasarkan Timestamp
point_clouds_grouped, labels_grouped, timestamps_grouped = group_point_clouds_by_timestamp(scaled_data, features_to_scale)
del scaled_data; gc.collect()
if point_clouds_grouped is None: exit()
n_features_detected = 4
if len(point_clouds_grouped) > 0 and isinstance(point_clouds_grouped[0], np.ndarray): n_features_detected = point_clouds_grouped[0].shape[1]
print(f"Jumlah fitur terdeteksi per titik: {n_features_detected}")

# 4. Pad Semua Frame
X_padded = pad_all_frames(point_clouds_grouped, MAX_POINTS)
del point_clouds_grouped; gc.collect()

# 5. Buat Window
X_windows, y_windows = create_windowed_data_from_padded(X_padded, labels_grouped, timestamps_grouped, WINDOW_SIZE, STEP)
del X_padded, labels_grouped, timestamps_grouped; gc.collect()
if X_windows is None: exit()

# 6. Split Train/Test
X_train_windows, X_test_windows, y_train_labels, y_test_labels = train_test_split( X_windows, y_windows, test_size=0.25, random_state=42, stratify=y_windows)
del X_windows, y_windows; gc.collect()
print(f"Data siap: Train={X_train_windows.shape}, Test/Validation={X_test_windows.shape}")

# 7. Augmentasi Data Training
X_train_augmented, y_train_augmented = augment_windowed_data(X_train_windows, y_train_labels)
del X_train_windows, y_train_labels; gc.collect()

# --- Inisialisasi & Kompilasi Model (Di dalam Strategy Scope) ---
print("--- Inisialisasi & Kompilasi Model (dalam Strategy Scope) ---")
with strategy.scope():
    model = PointCNN_Windowed( num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, max_points=MAX_POINTS, num_features=n_features_detected )
    print(f"Kompilasi model dengan optimizer AdamW: LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}.")
    model.compile( optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY), loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
    try: model.build(input_shape=(None, WINDOW_SIZE, MAX_POINTS, n_features_detected)); model.summary(expand_nested=True)
    except Exception as e: print(f"Peringatan: Tidak dapat build/summarize model: {e}")
print("-" * 30)

# 9. Persiapan Dataset TensorFlow
print("--- Persiapan Dataset ---")
print(f"Membuat tf.data Datasets dengan Global Batch Size={GLOBAL_BATCH_SIZE}...")
train_tf_ds = tf.data.Dataset.from_tensor_slices((X_train_augmented, y_train_augmented))
train_tf_ds = train_tf_ds.shuffle(buffer_size=len(X_train_augmented)).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_tf_ds = tf.data.Dataset.from_tensor_slices((X_test_windows, y_test_labels))
val_tf_ds = val_tf_ds.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print("tf.data Datasets dibuat.")
del X_train_augmented, y_train_augmented; gc.collect()
print("-" * 30)

# 10. Definisikan Callbacks
print("--- Definisi Callbacks ---")
callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1, mode='min'),
    ModelCheckpoint(filepath=MODEL_BEST_LOSS_PATH, monitor='val_loss', save_best_only=True, verbose=1, mode='min'),
    ModelCheckpoint(filepath=MODEL_BEST_ACC_PATH, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1, mode='min'),
    TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
]
print(f"Callbacks didefinisikan. Memonitor 'val_loss' dan 'val_accuracy'.")
print(f"Path Model Loss Terbaik: {MODEL_BEST_LOSS_PATH}")
print(f"Path Model Akurasi Terbaik: {MODEL_BEST_ACC_PATH}")
print("-" * 30)

# 11. Latih Model
print("--- Pelatihan Model ---")
print(f"Memulai pelatihan model hingga {EPOCHS} epochs menggunakan {strategy.num_replicas_in_sync} replika...")
history = model.fit( train_tf_ds, epochs=EPOCHS, validation_data=val_tf_ds, callbacks=callbacks_list, verbose=1 )
print("Pelatihan model selesai.")
print("-" * 30)

# 12. Evaluasi Akhir & Pelaporan
def evaluate_and_report(model_instance, model_name, test_data_np, test_labels_np, output_dir, class_names_list):
    """ Mengevaluasi model dan menyimpan laporan serta confusion matrix. """
    print(f"\n--- Mengevaluasi Model: {model_name} ---")
    eval_loss, eval_acc = model_instance.evaluate(test_data_np, test_labels_np, verbose=0, batch_size=GLOBAL_BATCH_SIZE)
    print(f'{model_name} - Test Loss: {eval_loss:.4f}'); print(f'{model_name} - Test Accuracy: {eval_acc*100:.2f}%')
    print(f"Menghasilkan prediksi untuk {model_name}...");
    y_pred_probs = model_instance.predict(test_data_np, batch_size=GLOBAL_BATCH_SIZE)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    print(f"\nClassification Report ({model_name}):");
    report = classification_report(test_labels_np, y_pred_classes, target_names=class_names_list, digits=4)
    print(report)
    report_filename = os.path.join(output_dir, f'classification_report_{model_name.replace(" ", "_").lower()}.txt')
    try:
        with open(report_filename, 'w') as f: f.write(f"Metrik Evaluasi ({model_name}):\nTest Loss: {eval_loss:.4f}\nTest Accuracy: {eval_acc*100:.2f}%\n\nClassification Report:\n{report}")
        print(f"Classification report disimpan ke {report_filename}")
    except Exception as e: print(f"Peringatan: Tidak dapat menyimpan classification report: {e}")
    print(f"Membuat Confusion Matrix untuk {model_name}...");
    cm = confusion_matrix(test_labels_np, y_pred_classes)
    plt.figure(figsize=(8, 6));
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_list, yticklabels=class_names_list, annot_kws={"size": 12})
    plt.xlabel('Predicted Label', fontsize=12); plt.ylabel('True Label', fontsize=12); plt.title(f'Confusion Matrix ({model_name})\nAccuracy: {eval_acc*100:.2f}%', fontsize=14)
    plt.xticks(rotation=0); plt.yticks(rotation=0); plt.tight_layout()
    cm_filename = os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    try:
        plt.savefig(cm_filename); print(f"Confusion Matrix disimpan ke {cm_filename}")
    except Exception as e: print(f"Peringatan: Tidak dapat menyimpan plot confusion matrix: {e}")
    plt.close()

print("--- Evaluasi Model Final ---")
custom_objects = {'XConvLayer_PointCNN': XConvLayer_PointCNN, 'PointCNN_Windowed': PointCNN_Windowed}
print(f"\nMemuat model loss validasi terbaik dari: {MODEL_BEST_LOSS_PATH}")
if os.path.exists(MODEL_BEST_LOSS_PATH):
    try:
        model_loss = models.load_model(MODEL_BEST_LOSS_PATH, custom_objects=custom_objects)
        print("Model loss validasi terbaik berhasil dimuat.")
        evaluate_and_report(model_loss, "Best Validation Loss", X_test_windows, y_test_labels, GRAPHICS_DIR, Figure_Name)
    except Exception as e:
        print(f"ERROR: Gagal memuat atau mengevaluasi model loss terbaik dari {MODEL_BEST_LOSS_PATH}: {e}")
        print("Mengevaluasi model langsung dari memori...")
        evaluate_and_report(model, "Best Validation Loss (from memory)", X_test_windows, y_test_labels, GRAPHICS_DIR, Figure_Name)
else:
    print(f"PERINGATAN: File checkpoint untuk model loss terbaik tidak ditemukan di {MODEL_BEST_LOSS_PATH}. Mengevaluasi model dari memori.")
    evaluate_and_report(model, "Best Validation Loss (from memory)", X_test_windows, y_test_labels, GRAPHICS_DIR, Figure_Name)

print(f"\nMemuat model akurasi validasi terbaik dari: {MODEL_BEST_ACC_PATH}")
if os.path.exists(MODEL_BEST_ACC_PATH):
    try:
        model_acc = models.load_model(MODEL_BEST_ACC_PATH, custom_objects=custom_objects)
        print("Model akurasi validasi terbaik berhasil dimuat.")
        evaluate_and_report(model_acc, "Best Validation Accuracy", X_test_windows, y_test_labels, GRAPHICS_DIR, Figure_Name)
    except Exception as e:
        print(f"ERROR: Gagal memuat atau mengevaluasi model akurasi terbaik dari {MODEL_BEST_ACC_PATH}: {e}")
else:
    print(f"PERINGATAN: File checkpoint untuk model akurasi terbaik tidak ditemukan di {MODEL_BEST_ACC_PATH}. Melewati evaluasi.")
print("-" * 30)


# 13. Visualisasi Riwayat Training Final
print("--- Visualisasi Riwayat Training Final ---")
try:
    history_dict = history.history; epochs_range = range(1, len(history_dict['loss']) + 1)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1); plt.plot(epochs_range, history_dict['accuracy'], label='Training Accuracy'); plt.plot(epochs_range, history_dict['val_accuracy'], label='Validation Accuracy')
    best_acc_epoch_final = np.argmax(history_dict['val_accuracy']); plt.scatter(best_acc_epoch_final + 1, history_dict['val_accuracy'][best_acc_epoch_final], marker='o', color='purple', label=f'Best Val Acc (Epoch {best_acc_epoch_final+1})', zorder=5, s=80)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(loc='lower right'); plt.title('Final Model Accuracy'); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(epochs_range, history_dict['loss'], label='Training Loss'); plt.plot(epochs_range, history_dict['val_loss'], label='Validation Loss')
    best_loss_epoch_final = np.argmin(history_dict['val_loss']); plt.scatter(best_loss_epoch_final + 1, history_dict['val_loss'][best_loss_epoch_final], marker='o', color='red', label=f'Best Val Loss (Epoch {best_loss_epoch_final+1})', zorder=5, s=80)
    plt.scatter(best_acc_epoch_final + 1, history_dict['val_loss'][best_acc_epoch_final], marker='x', color='purple', label=f'Loss at Best Acc Epoch', zorder=5, s=80)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(loc='upper right'); plt.title('Final Model Loss'); plt.grid(True)
    plt.suptitle(f'Final Model Training History ({RUN_NAME})', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    history_plot_path = os.path.join(GRAPHICS_DIR, 'final_training_curves.png'); plt.savefig(history_plot_path); plt.close()
    print(f"Plot riwayat training final disimpan ke {history_plot_path}")
except Exception as e: print(f"Peringatan: Tidak dapat memplot riwayat training final: {e}")
print("-" * 30)

# 14. Ringkasan Akhir
print("--- Ringkasan Akhir ---")
print(f"Eksekusi skrip selesai untuk run: {RUN_NAME}")
print(f"Hasil disimpan di direktori: {OUTPUT_BASE_DIR}")
print(f"Log di: {LOG_DIR}")
print(f"Model terbaik (berdasarkan val_loss) disimpan di: {MODEL_BEST_LOSS_PATH}")
print(f"Model terbaik (berdasarkan val_accuracy) disimpan di: {MODEL_BEST_ACC_PATH}")
print(f"Scaler disimpan di: {SCALER_PATH}") # <<< Info path scaler ditambahkan >>>
print("Tinjau grafik dan laporan yang disimpan untuk perbandingan performa detail.")
print("-" * 30)

# Hapus data test dari memori jika perlu
del X_test_windows, y_test_labels; gc.collect()