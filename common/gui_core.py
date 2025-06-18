# General Library Imports
import json
import time
from serial.tools import list_ports
import os
import sys
from contextlib import suppress

# PyQt Imports
from PySide2 import QtGui
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QKeySequence
from PySide2.QtWidgets import (
    QAction,
    QTabWidget,
    QGridLayout,
    QMenu,
    QGroupBox,
    QLineEdit,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QWidget,
    QShortcut,
    QSlider,
    QCheckBox,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSpacerItem,
    QSizePolicy
)

# Local Imports
from cached_data import CachedDataType
from demo_defines import *
from gui_threads import *
from parseFrame import parseStandardFrame
from Common_Tabs.plot_1d import Plot1D
from Common_Tabs.plot_2d import Plot2D
from Common_Tabs.plot_3d import Plot3D
from Demo_Classes.people_tracking import PeopleTracking

# Logger
import logging
log = logging.getLogger(__name__)

class Window(QMainWindow):
    def __init__(self, parent=None, size=[], title="Human and Vehicle Classification"):
        super(Window, self).__init__(parent)
        log.info("Inisialisasi Window GUI")
        self.core = Core()
        self.core.window = self  # Simpan referensi ke window di core
        self.setWindowIcon(QtGui.QIcon("./images/logo.png"))
        self.shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        self.shortcut.activated.connect(self.close)

        # Set the layout
        # Create tab for different graphing options
        self.demoTabs = QTabWidget()
        
        # Inisialisasi gridLayout terlebih dahulu untuk kompatibilitas
        self.gridLayout = QGridLayout()
        
        # Buat layout untuk panel kiri
        self.leftPanelLayout = QVBoxLayout()
        self.leftPanelLayout.setSpacing(5)  # Kurangi spacing vertikal
        self.leftPanelLayout.setContentsMargins(5, 5, 5, 5)  # Kurangi margin
        
        # Add connect options
        log.info("Inisialisasi panel konfigurasi")
        self.initConfigPane()
        log.info("Inisialisasi panel koneksi")
        self.initConnectionPane()
        
        # Tambahkan spacer antara comBox dan configBox
        verticalSpacer = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)
        
        self.leftPanelLayout.addWidget(self.comBox)
        self.leftPanelLayout.addItem(verticalSpacer)  # Tambahkan spacer
        self.leftPanelLayout.addWidget(self.configBox)
        self.leftPanelLayout.addStretch(1)  # Tambahkan spacer di bagian bawah
        
        # Buat widget untuk panel kiri
        leftPanelWidget = QWidget()
        leftPanelWidget.setLayout(self.leftPanelLayout)
        leftPanelWidget.setFixedWidth(320)  # Perlebar panel kiri
        
        # Buat panel klasifikasi di sebelah kanan
        log.info("Inisialisasi panel klasifikasi")
        self.initClassificationPane()
        
        # Buat splitter dan tambahkan panel kiri, demoTabs, dan panel klasifikasi
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(leftPanelWidget)
        self.splitter.addWidget(self.demoTabs)
        self.splitter.addWidget(self.classificationBox)
        
        # Atur proporsi awal (lebih lebar untuk panel kiri dan kanan)
        self.splitter.setSizes([320, 400, 320])
        
        # Layout utama
        self.mainLayout = QGridLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)  # Hapus margin
        self.mainLayout.addWidget(self.splitter, 0, 0, 1, 1)
        
        # Replay slider
        self.core.sl.setMinimum(0)
        self.core.sl.setMaximum(30)
        self.core.sl.setValue(20)
        self.core.sl.setTickPosition(QSlider.TicksBelow)
        self.core.sl.setTickInterval(5)

        self.replayBox = QGroupBox("Replay")
        self.replayLayout = QGridLayout()
        self.replayLayout.addWidget(self.core.sl, 0, 0, 1, 1)
        self.replayBox.setLayout(self.replayLayout)
        self.replayBox.setVisible(False)
        
        self.mainLayout.addWidget(self.replayBox, 1, 0, 1, 1)

        self.central = QWidget()
        self.central.setLayout(self.mainLayout)
        self.setWindowTitle(title)

        log.info("Inisialisasi menu bar")
        self.initMenuBar()
        self.core.replay = False
        self.setCentralWidget(self.central)
        log.info("GUI berhasil diinisialisasi")
        self.showMaximized()

    def initMenuBar(self):
        menuBar = self.menuBar()
        # Creating menus using a QMenu object
        fileMenu = QMenu("&File", self)
        playbackMenu = QMenu("&Playback", self)
        self.logOutputAction = QAction("Log Terminal Output to File", self)
        self.playbackAction = QAction("Load and Replay", self)
        self.playbackAction.triggered.connect(self.loadForReplay)
        self.playbackAction.setCheckable(True)
        self.logOutputAction.triggered.connect(self.toggleLogOutput)
        self.logOutputAction.setCheckable(True)
        playbackMenu.addAction(self.playbackAction)
        fileMenu.addAction(self.logOutputAction)
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(playbackMenu)

    def initClassificationPane(self):
        # Tambahkan panel klasifikasi
        self.classificationBox = QGroupBox("Hasil Klasifikasi")
        self.classificationBox.setStyleSheet("""
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                font-size: 14px;
                background-color: #F8F8F8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #F8F8F8;
            }
        """)
        self.classificationBox.setFixedWidth(320)  # Perlebar panel klasifikasi
        
        self.classificationLayout = QVBoxLayout()
        self.classificationLayout.setSpacing(2)  # Kurangi spacing vertikal
        self.classificationLayout.setContentsMargins(10, 10, 10, 10)  # Atur margin
        
        # Tambahkan header
        self.headerLabel = QLabel("Deteksi Objek")
        self.headerLabel.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 5px; text-align: center;")
        self.headerLabel.setAlignment(Qt.AlignCenter)
        
        # Tambahkan indikator status di tengah
        self.statusIndicator = QLabel()
        self.statusIndicator.setMinimumSize(60, 60)
        self.statusIndicator.setMaximumSize(60, 60)
        self.statusIndicator.setStyleSheet("background-color: gray; border-radius: 30px;")
        self.statusIndicator.setAlignment(Qt.AlignCenter)
        
        # Tambahkan container untuk indikator
        indicatorContainer = QWidget()
        indicatorLayout = QHBoxLayout()
        indicatorLayout.setContentsMargins(0, 0, 0, 0)
        indicatorLayout.addStretch()
        indicatorLayout.addWidget(self.statusIndicator)
        indicatorLayout.addStretch()
        indicatorContainer.setLayout(indicatorLayout)
        
        # Label untuk menampilkan hasil klasifikasi
        self.classLabel = QLabel("Kelas: -")
        self.confidenceLabel = QLabel("Confidence: -")
        self.classLabel.setStyleSheet("font-size: 14pt; font-weight: bold; margin-top: 5px;")
        self.confidenceLabel.setStyleSheet("font-size: 12pt; margin-bottom: 5px;")
        self.classLabel.setAlignment(Qt.AlignCenter)
        self.confidenceLabel.setAlignment(Qt.AlignCenter)
        
        # Tambahkan deskripsi kelas
        self.descriptionLabel = QLabel("Deskripsi:")
        self.descriptionLabel.setStyleSheet("font-size: 12pt; font-weight: bold; margin-top: 10px;")
        self.descriptionText = QLabel("Menunggu deteksi...")
        self.descriptionText.setStyleSheet("font-size: 11pt;")
        self.descriptionText.setWordWrap(True)
        
        # Tambahkan semua ke layout utama
        self.classificationLayout.addWidget(self.headerLabel)
        self.classificationLayout.addWidget(indicatorContainer)
        self.classificationLayout.addWidget(self.classLabel)
        self.classificationLayout.addWidget(self.confidenceLabel)
        self.classificationLayout.addWidget(self.descriptionLabel)
        self.classificationLayout.addWidget(self.descriptionText)
        
        # Tambahkan vertical spacer setelah deskripsi
        verticalSpacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.classificationLayout.addItem(verticalSpacer)
        
        # Tambahkan separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
        self.classificationLayout.addWidget(separator)
        
        # Tambahkan panel statistik
        self.statsGroup = QGroupBox("Statistics")
        self.statsGroup.setStyleSheet("""
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                margin-top: 1ex;
                font-weight: bold;
                font-size: 12px;
                background-color: #F0F0F0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                background-color: #F0F0F0;
            }
        """)
        self.statsLayout = QVBoxLayout()
        self.statsLayout.setSpacing(2)  # Kurangi spacing
        self.statsLayout.setContentsMargins(5, 10, 5, 5)  # Kurangi margin
        
        self.frameLabel = QLabel("Frame: 0")
        self.plotTimeLabel = QLabel("Plot Time: 0 ms")
        self.pointsLabel = QLabel("Points: 0")
        self.targetsLabel = QLabel("Targets: 0")
        self.powerLabel = QLabel("Average Power: 0 mW")
        
        self.statsLayout.addWidget(self.frameLabel)
        self.statsLayout.addWidget(self.plotTimeLabel)
        self.statsLayout.addWidget(self.pointsLabel)
        self.statsLayout.addWidget(self.targetsLabel)
        self.statsLayout.addWidget(self.powerLabel)
        
        self.statsGroup.setLayout(self.statsLayout)
        self.classificationLayout.addWidget(self.statsGroup)
        
        # Tambahkan panel plot controls
        self.plotControlsGroup = QGroupBox("Plot Controls")
        self.plotControlsGroup.setStyleSheet("""
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                margin-top: 1ex;
                font-weight: bold;
                font-size: 12px;
                background-color: #F0F0F0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                background-color: #F0F0F0;
            }
        """)
        self.plotControlsLayout = QGridLayout()
        self.plotControlsLayout.setSpacing(2)  # Kurangi spacing
        self.plotControlsLayout.setContentsMargins(5, 10, 5, 5)  # Kurangi margin
        
        self.colorPointsLabel = QLabel("Color Points By:")
        self.colorPointsCombo = QComboBox()
        self.colorPointsCombo.addItems(["SNR", "Height", "Doppler", "Associated Track"])
        self.colorPointsCombo.setCurrentText("Doppler")  # Set default ke Doppler
        self.colorPointsCombo.currentTextChanged.connect(self.onColorPointsChanged)
        
        self.plotControlsLayout.addWidget(self.colorPointsLabel, 0, 0)
        self.plotControlsLayout.addWidget(self.colorPointsCombo, 0, 1)
        
        self.plotControlsGroup.setLayout(self.plotControlsLayout)
        self.classificationLayout.addWidget(self.plotControlsGroup)
        
        self.classificationLayout.addStretch(1)
        self.classificationBox.setLayout(self.classificationLayout)
    
    def onColorPointsChanged(self, text):
        log.info(f"Mengubah mode warna point cloud menjadi: {text}")
        if text == "Doppler":
            self.core.setPointColorMode(COLOR_MODE_DOPPLER)
        elif text == "SNR":
            self.core.setPointColorMode(COLOR_MODE_SNR)
        elif text == "Height":
            self.core.setPointColorMode(COLOR_MODE_HEIGHT)
        elif text == "Associated Track":
            self.core.setPointColorMode(COLOR_MODE_TRACK)

    def loadForReplay(self, state):
        if (state):
            log.info("Memulai mode replay")
            self.recordAction.setChecked(False)
            self.core.replayFile = QFileDialog.getOpenFileName(self, 'Open Replay JSON File', '.',"JSON Files (*.json)")
            log.info(f"File replay dipilih: {self.core.replayFile[0]}")
            self.core.replay = True
            self.core.loadForReplay(True)
            # Disable COM Ports/Device/Demo/Config
            self.deviceList.setEnabled(False)
            self.cliCom.setEnabled(False)
            self.dataCom.setEnabled(False)
            self.connectButton.setEnabled(False)
            self.filename_edit.setEnabled(False)
            self.selectConfig.setEnabled(False)
            self.sendConfig.setEnabled(False)
            self.start.setEnabled(True)
            self.start.setText("Replay")
            self.replayBox.setVisible(True)
        else:
            log.info("Menghentikan mode replay")
            self.core.replay = False
            # Enable COM Ports/Device/Demo/Config
            self.deviceList.setEnabled(True)
            self.cliCom.setEnabled(True)
            self.dataCom.setEnabled(True)
            self.connectButton.setEnabled(True)
            self.filename_edit.setEnabled(True)
            self.selectConfig.setEnabled(True)
            self.sendConfig.setEnabled(True)
            self.start.setText("Start without Send Configuration")
            self.replayBox.setVisible(False)

    def toggleSaveData(self):
        if self.recordAction.isChecked():
            log.info("Mengaktifkan penyimpanan data")
            self.core.parser.setSaveBinary(True)
        else:
            log.info("Menonaktifkan penyimpanan data")
            self.core.parser.setSaveBinary(False)
            self.core.replay = False
            # Enable COM Ports/Device/Demo/Config
            self.deviceList.setEnabled(True)
            self.cliCom.setEnabled(True)
            self.dataCom.setEnabled(True)
            self.connectButton.setEnabled(True)
            self.filename_edit.setEnabled(True)
            self.selectConfig.setEnabled(True)
            self.start.setText("Start without Send Configuration")

    def toggleLogOutput(self):
        if (self.logOutputAction.isChecked()):
            # Save terminal output to logFile, set 0 to show terminal output
            ts = time.localtime()
            terminalFileName = str(
                "logfile_"
                + str(ts[2])
                + str(ts[1])
                + str(ts[0])
                + "_"
                + str(ts[3])
                + str(ts[4])
                + ".txt"
            )
            log.info(f"Mengalihkan output terminal ke file: {terminalFileName}")
            sys.stdout = open(terminalFileName, "w")
        else:
            log.info("Mengembalikan output terminal ke konsol")
            sys.stdout = sys.__stdout__

    def initConnectionPane(self):
        self.comBox = QGroupBox("Connect to COM Ports")
        self.comBox.setStyleSheet("""
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                margin-top: 1ex;
                font-weight: bold;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)
        
        # Atur lebar yang sama untuk semua input box
        fixed_width = 150  # Atur lebar yang diinginkan
        
        self.cliCom = QLineEdit("")
        self.cliCom.setFixedWidth(fixed_width)
        
        self.dataCom = QLineEdit("")
        self.dataCom.setFixedWidth(fixed_width)
        
        self.connectStatus = QLabel("Not Connected")
        self.connectButton = QPushButton("Connect")
        self.connectButton.clicked.connect(self.onConnect)
        
        self.deviceList = QComboBox()
        self.deviceList.setFixedWidth(fixed_width)
        
        self.recordAction = QCheckBox("Save Data to File", self)
        
        # Hanya gunakan device IWR6843
        self.deviceList.addItem("xWR6843")
        self.deviceList.currentIndexChanged.connect(self.onChangeDevice)

        self.comLayout = QVBoxLayout()  # Gunakan VBoxLayout untuk mengurangi spacing vertikal
        self.comLayout.setSpacing(2)  # Kurangi spacing vertikal
        self.comLayout.setContentsMargins(5, 10, 5, 5)  # Kurangi margin
        
        # Buat layout untuk device
        deviceLayout = QHBoxLayout()
        deviceLayout.addWidget(QLabel("Device:"))
        deviceLayout.addWidget(self.deviceList)
        
        # Buat layout untuk CLI COM
        cliLayout = QHBoxLayout()
        cliLayout.addWidget(QLabel("CLI COM:"))
        cliLayout.addWidget(self.cliCom)
        
        # Buat layout untuk DATA COM
        dataLayout = QHBoxLayout()
        dataLayout.addWidget(QLabel("DATA COM:"))
        dataLayout.addWidget(self.dataCom)
        
        # Buat layout untuk connect button dan status
        connectLayout = QHBoxLayout()
        connectLayout.addWidget(self.connectButton)
        connectLayout.addWidget(self.connectStatus)
        
        # Tambahkan semua layout ke layout utama
        self.comLayout.addLayout(deviceLayout)
        self.comLayout.addLayout(cliLayout)
        self.comLayout.addLayout(dataLayout)
        self.comLayout.addLayout(connectLayout)
        
        # Tambahkan checkbox
        self.recordAction.stateChanged.connect(self.toggleSaveData)
        self.comLayout.addWidget(self.recordAction)
        
        self.comBox.setLayout(self.comLayout)

        # Find all Com Ports
        log.info("Mencari port COM yang tersedia...")
        serialPorts = list(list_ports.comports())
        # Find default CLI Port and Data Port
        for port in serialPorts:
            if (
                CLI_XDS_SERIAL_PORT_NAME in port.description
                or CLI_SIL_SERIAL_PORT_NAME in port.description
            ):
                log.info(f"CLI COM Port found: {port.device}")
                comText = port.device
                comText = comText.replace("COM", "")
                self.cliCom.setText(comText)
            elif (
                DATA_XDS_SERIAL_PORT_NAME in port.description
                or DATA_SIL_SERIAL_PORT_NAME in port.description
            ):
                log.info(f"Data COM Port found: {port.device}")
                comText = port.device
                comText = comText.replace("COM", "")
                self.dataCom.setText(comText)

        self.core.isGUILaunched = 1
        self.loadCachedData()

    def initConfigPane(self):
        self.configBox = QGroupBox("Configuration")
        self.configBox.setStyleSheet("""
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                margin-top: 10px;  /* Tambahkan margin atas yang lebih besar */
                font-weight: bold;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)
        
        self.selectConfig = QPushButton("Select Configuration")
        self.sendConfig = QPushButton("Start and Send Configuration")
        self.start = QPushButton("Start without Send Configuration")
        self.sensorStop = QPushButton("Send sensorStop Command")
        self.sensorStop.setToolTip("Stop sensor (only works if lowPowerCfg is 0)")
        self.filename_edit = QLineEdit()
        self.selectConfig.clicked.connect(lambda: self.selectCfg(self.filename_edit))
        self.sendConfig.setEnabled(False)
        self.start.setEnabled(False)
        self.sendConfig.clicked.connect(self.sendCfg)
        self.start.clicked.connect(self.startApp)
        self.sensorStop.clicked.connect(self.stopSensor)
        self.sensorStop.setHidden(True)

        self.configLayout = QVBoxLayout()  # Gunakan VBoxLayout untuk mengurangi spacing vertikal
        self.configLayout.setSpacing(2)  # Kurangi spacing vertikal
        self.configLayout.setContentsMargins(5, 10, 5, 5)  # Kurangi margin
        
        # Buat layout untuk file selection
        fileLayout = QHBoxLayout()
        fileLayout.addWidget(self.filename_edit)
        fileLayout.addWidget(self.selectConfig)
        
        # Tambahkan semua ke layout utama
        self.configLayout.addLayout(fileLayout)
        self.configLayout.addWidget(self.sendConfig)
        self.configLayout.addWidget(self.start)
        self.configLayout.addWidget(self.sensorStop)
        
        self.configBox.setLayout(self.configLayout)

    def loadCachedData(self):
        log.info("Memuat data yang tersimpan dari cache")
        self.core.loadCachedData(
            self.deviceList, self.recordAction, self.gridLayout, self.demoTabs
        )

    # Callback function when device is changed
    def onChangeDevice(self):
        log.info(f"Mengubah device menjadi: {self.deviceList.currentText()}")
        self.core.changeDevice(
            self.deviceList, self.gridLayout, self.demoTabs
        )
        self.core.updateCOMPorts(self.cliCom, self.dataCom)
        self.core.updateResetButton(self.sensorStop)

    # Callback function when connect button clicked
    def onConnect(self):
        if (self.connectStatus.text() == "Not Connected" or self.connectStatus.text() == "Unable to Connect"):
            log.info(f"Mencoba menghubungkan ke CLI COM: {self.cliCom.text()} dan DATA COM: {self.dataCom.text()}")
            if self.core.connectCom(self.cliCom, self.dataCom, self.connectStatus) == 0:
                log.info("Berhasil terhubung ke port COM")
                self.connectButton.setText("Reset Connection")
                self.sendConfig.setEnabled(True)
                self.start.setEnabled(True)
            else:
                log.error("Gagal terhubung ke port COM")
                self.sendConfig.setEnabled(False)
                self.start.setEnabled(False)
        else:
            log.info("Mereset koneksi port COM")
            self.core.gracefulReset()
            self.connectButton.setText("Connect")
            self.connectStatus.setText("Not Connected")
            self.sendConfig.setEnabled(False)
            self.start.setEnabled(False)

    # Callback function when 'Select Configuration' is clicked
    def selectCfg(self, filename):
        log.info("Memilih file konfigurasi")
        self.core.selectCfg(filename)

    # Callback function when 'Start and Send Configuration' is clicked
    def sendCfg(self):
        log.info("Mengirim konfigurasi ke device")
        self.core.sendCfg()

    # Callback function to send sensorStop to device
    def stopSensor(self):
        log.info("Mengirim perintah sensorStop ke device")
        self.core.stopSensor()

    # Callback function when 'Start without Send Configuration' is clicked
    def startApp(self):
        if (self.core.replay and self.core.playing is False):
            log.info("Memulai replay")
            self.start.setText("Pause")
        elif (self.core.replay and self.core.playing is True):
            log.info("Menjeda replay")
            self.start.setText("Replay")
        else:
            log.info("Memulai aplikasi tanpa mengirim konfigurasi")
        self.core.startApp()

class Core:
    def __init__(self):
        log.info("Inisialisasi Core")
        self.cachedData = CachedDataType()
        self.device = "xWR6843"
        self.demo = DEMO_3D_PEOPLE_TRACKING
        self.frameTime = 50
        self.parser = UARTParser(type="DoubleCOMPort")
        self.replayFile = "replay.json"
        self.replay = False
        # set to 1
        self.isGUILaunched = 0
        self.sl = QSlider(Qt.Horizontal)
        self.sl.valueChanged.connect(self.sliderValueChange)
        self.playing = False
        self.replayFrameNum = 0
        
        # Inisialisasi thread klasifikasi
        self.classification_thread = None
        self.classification_result = None
        self.window = None  # Referensi ke window untuk update UI

        # Populated with each demo and it's corresponding object
        self.demoClassDict = {
            DEMO_3D_PEOPLE_TRACKING: PeopleTracking(),
        }
        log.info("Core berhasil diinisialisasi")

    def loadCachedData(self, deviceList, recordAction, gridLayout, demoTabs):
        deviceName = self.cachedData.getCachedDeviceName()
        recordState = bool(self.cachedData.getCachedRecord())
        if deviceName in self.getDeviceList():
            deviceList.setCurrentIndex(self.getDeviceList().index(deviceName))
        self.changeDevice(deviceList, gridLayout, demoTabs)
        if recordState:
            recordAction.setChecked(True)
        log.info(f"Data cache berhasil dimuat: Device={deviceName}, Record={recordState}")

    def getDemoList(self):
        return [DEMO_3D_PEOPLE_TRACKING]

    def getDeviceList(self):
        return ["xWR6843"]

    def changeDevice(self, deviceList, gridLayout, demoTabs):
        self.device = deviceList.currentText()
        if (self.isGUILaunched):
            self.cachedData.setCachedDeviceName(self.device)

        self.parser.parserType = "DoubleCOMPort"
        log.info(f"Mengubah device menjadi {self.device} dengan parser type {self.parser.parserType}")

        permanentWidgetsList = ["Connect to COM Ports", "Configuration", "Tabs", "Replay", "Hasil Klasifikasi"]
        # Destroy current contents of graph pane
        for _ in range(demoTabs.count()):
            demoTabs.removeTab(0)

        for i in range(gridLayout.count()):
            try:
                currWidget = gridLayout.itemAt(i).widget()
                if currWidget.title() not in permanentWidgetsList:
                    currWidget.setVisible(False)
            except AttributeError as e:
                log.log(0, "Demo Tabs don't have title attribute. This is OK")
                continue

        # Make call to selected demo's initialization function
        log.info(f"Menginisialisasi GUI untuk demo {self.demo}")
        self.demoClassDict[self.demo].setupGUI(gridLayout, demoTabs, self.device)

    def updateCOMPorts(self, cliCom, dataCom):
        dataCom.setEnabled(True)
        log.info("Port COM diperbarui")

    def updateResetButton(self, sensorStopButton):
        sensorStopButton.setHidden(True)

    def stopSensor(self):
        log.info("Mengirim perintah sensorStop 0 ke device")
        self.parser.sendLine("sensorStop 0")

    def selectFile(self, filename):
        try:
            current_dir = os.getcwd()
            configDirectory = current_dir
            path = self.cachedData.getCachedCfgPath()
            if path != "":
                configDirectory = path
        except:
            configDirectory = ""

        log.info(f"Membuka dialog pemilihan file konfigurasi dari direktori: {configDirectory}")
        fname = QFileDialog.getOpenFileName(caption="Open .cfg File", dir=configDirectory, filter="cfg(*.cfg)")
        filename.setText(str(fname[0]))
        log.info(f"File konfigurasi dipilih: {fname[0]}")
        return fname[0]

    def parseCfg(self, fname):
        log.info(f"Parsing file konfigurasi: {fname}")
        if (self.replay):
            self.cfg = self.data['cfg']
            log.info("Menggunakan konfigurasi dari file replay")
        else:
            with open(fname, "r") as cfg_file:
                self.cfg = cfg_file.readlines()
            log.info(f"Membaca {len(self.cfg)} baris dari file konfigurasi")

        self.parser.cfg = self.cfg
        self.parser.demo = self.demo
        self.parser.device = self.device

        for line in self.cfg:
            args = line.split()
            if len(args) > 0:
                # trackingCfg
                if args[0] == "trackingCfg":
                    if len(args) < 5:
                        log.error("trackingCfg had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseTrackingCfg(args)
                            log.info(f"Parsing trackingCfg: {args}")

                elif args[0] == "SceneryParam" or args[0] == "boundaryBox":
                    if len(args) < 7:
                        log.error(
                            "SceneryParam/boundaryBox had fewer arguments than expected"
                        )
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseBoundaryBox(args)
                            log.info(f"Parsing boundaryBox: {args}")

                elif args[0] == "frameCfg":
                    if len(args) < 4:
                        log.error("frameCfg had fewer arguments than expected")
                    else:
                        self.frameTime = float(args[5]) / 2
                        log.info(f"Parsing frameCfg: frameTime set to {self.frameTime} ms")

                elif args[0] == "sensorPosition":
                    # sensorPosition for x843 family has 3 args
                    if len(args) < 4:
                        log.error("sensorPosition had fewer arguments than expected")
                    else:
                        with suppress(AttributeError):
                            self.demoClassDict[self.demo].parseSensorPosition(
                                args, True
                            )
                            log.info(f"Parsing sensorPosition: {args}")

        # Initialize 1D plot values based on cfg file
        with suppress(AttributeError):
            self.demoClassDict[self.demo].setRangeValues()
            log.info("Range values diinisialisasi berdasarkan file konfigurasi")

        log.info("Parsing file konfigurasi selesai")

    def selectCfg(self, filename):
        try:
            file = self.selectFile(filename)
            self.cachedData.setCachedCfgPath(file) # cache the file and demo used
            self.parseCfg(file)
        except Exception as e:
            log.error(e)
            log.error(
                "Parsing .cfg file failed. Did you select a valid configuration file?"
            )
        log.debug("Demo Changed to " + self.demo)

    def sendCfg(self):
        try:
            if self.demo != "Replay":
                log.info("Mengirim konfigurasi ke device")
                self.parser.sendCfg(self.cfg)
            sys.stdout.flush()
            log.info(f"Memulai timer parsing dengan interval {self.frameTime} ms")
            self.parseTimer.start(int(self.frameTime)) # need this line
        except Exception as e:
            log.error(e)
            log.error("Parsing .cfg file failed. Did you select the right file?")

    def updateGraph(self, outputDict):
        # Teruskan point cloud ke thread klasifikasi jika ada
        if hasattr(self, 'classification_thread') and self.classification_thread is not None:
            if 'pointCloud' in outputDict and outputDict['pointCloud'] is not None:
                self.classification_thread.add_point_cloud(outputDict['pointCloud'])
        
        # Tambahkan hasil klasifikasi ke outputDict jika ada
        if hasattr(self, 'classification_result') and self.classification_result is not None:
            outputDict['classificationResult'] = self.classification_result
        
        # Update statistik di panel klasifikasi
        if hasattr(self, 'window') and self.window is not None:
            # Update frame count
            if 'frameNum' in outputDict:
                self.window.frameLabel.setText(f"Frame: {outputDict['frameNum']}")
            
            # Update plot time
            if 'plotTime' in outputDict:
                self.window.plotTimeLabel.setText(f"Plot Time: {outputDict['plotTime']:.2f} ms")
            
            # Update points count
            if 'pointCloud' in outputDict and outputDict['pointCloud'] is not None:
                num_points = outputDict['pointCloud'].shape[0] if outputDict['pointCloud'].size > 0 else 0
                self.window.pointsLabel.setText(f"Points: {num_points}")
            
            # Update targets count
            if 'targets' in outputDict and outputDict['targets'] is not None:
                num_targets = len(outputDict['targets']) if outputDict['targets'] is not None else 0
                self.window.targetsLabel.setText(f"Targets: {num_targets}")
            
            # Update average power (dummy value for now)
            self.window.powerLabel.setText(f"Average Power: 0 mW")
        
        # Teruskan ke demo untuk visualisasi
        self.demoClassDict[self.demo].updateGraph(outputDict)

    def updateClassificationResult(self, result):
        """Callback saat hasil klasifikasi diterima"""
        self.classification_result = result
        class_name = result.get('class', 'Tidak diketahui')
        confidence = result.get('confidence', 0.0)
        vote_percentage = result.get('vote_percentage', 0.0)
        class_id = result.get('class_id', -1)
        
        # SPECIAL DEBUG FOR MOTOR UI RECEIVED
        if class_id == 2:
            log.warning(f"🏍️ UI RECEIVED MOTOR RESULT: {result}")
        
        # Log berdasarkan tipe hasil
        if class_name == "Tidak ada objek":
            log.info("Tidak ada objek terdeteksi, atau jumlah titik tidak cukup")
        else:
            # Tambahkan info voting jika tersedia
            if vote_percentage > 0:
                log.info(f"Hasil klasifikasi: {class_name}, Confidence: {confidence:.2f}, Voting: {vote_percentage:.2f}")
            else:
                log.info(f"Hasil klasifikasi: {class_name}, Confidence: {confidence:.2f}")
        
        # Update label di panel klasifikasi
        if hasattr(self, 'window') and self.window is not None:
            # Update label kelas dan confidence
            self.window.classLabel.setText(f"Kelas: {class_name}")
            
            # SPECIAL DEBUG FOR MOTOR UI UPDATE
            if class_id == 2:
                log.warning(f"🏍️ UPDATING UI LABELS FOR MOTOR: {class_name}")
            
            # Jika ada objek, tampilkan confidence
            if class_name != "Tidak ada objek":
                # Tambahkan informasi voting jika tersedia
                if vote_percentage > 0:
                    self.window.confidenceLabel.setText(f"Confidence: {confidence:.2f} (Voting: {vote_percentage:.2f})")
                else:
                    self.window.confidenceLabel.setText(f"Confidence: {confidence:.2f}")
            else:
                self.window.confidenceLabel.setText("Confidence: -")
            
            # Ubah warna label dan teks deskripsi berdasarkan kelas
            if class_name == 'Manusia':
                color = "color: blue;"
                self.window.statusIndicator.setStyleSheet("background-color: blue; border-radius: 30px;")
                self.window.descriptionText.setText("Terdeteksi manusia di area pengawasan. Gerakan manusia sedang dipantau.")
            elif class_name == 'Mobil':
                color = "color: red;"
                self.window.statusIndicator.setStyleSheet("background-color: red; border-radius: 30px;")
                self.window.descriptionText.setText("Terdeteksi mobil di area pengawasan. Kendaraan berukuran besar sedang dipantau.")
            elif class_name == 'Motor':
                color = "color: green;"
                self.window.statusIndicator.setStyleSheet("background-color: green; border-radius: 30px;")
                self.window.descriptionText.setText("Terdeteksi motor di area pengawasan. Kendaraan berukuran kecil sedang dipantau.")
            elif class_name == 'Tidak ada objek':
                color = "color: #888888;"  # Abu-abu
                self.window.statusIndicator.setStyleSheet("background-color: #CCCCCC; border-radius: 30px;")
                self.window.descriptionText.setText("Tidak ada objek terdeteksi di area pengawasan.")
            else:
                color = "color: black;"
                self.window.statusIndicator.setStyleSheet("background-color: gray; border-radius: 30px;")
                self.window.descriptionText.setText("Menunggu deteksi objek yang valid...")
            
            self.window.classLabel.setStyleSheet(f"font-size: 14pt; font-weight: bold; {color}")

    def setPointColorMode(self, mode):
        for demo in self.demoClassDict.values():
            if hasattr(demo, "pointColorMode"):
                demo.pointColorMode = mode
        log.info(f"Mode warna point cloud diubah menjadi: {mode}")

    def initClassificationThread(self, model_path, scaler_path):
        """Inisialisasi thread klasifikasi"""
        try:
            log.info(f"Memulai pemuatan model dari {model_path} dan scaler dari {scaler_path}")
            self.classification_thread = ClassificationThread(model_path, scaler_path)
            log.info("Menghubungkan sinyal hasil klasifikasi antara thread klasifikasi dan core")
            self.classification_thread.result.connect(self.updateClassificationResult)
            self.classification_thread.start()
            log.info("Thread klasifikasi berhasil diinisialisasi dan model siap digunakan")
        except Exception as e:
            log.error(f"Error saat inisialisasi thread klasifikasi: {e}")
            self.classification_thread = None

    def connectCom(self, cliCom, dataCom, connectStatus):
        log.info(f"Mencoba menghubungkan ke CLI COM: {cliCom.text()} dan DATA COM: {dataCom.text()}")
        # init threads and timers
        self.uart_thread = parseUartThread(self.parser)
        log.info("Menghubungkan sinyal UART thread dengan fungsi updateGraph")
        self.uart_thread.fin.connect(self.updateGraph)
        self.parseTimer = QTimer()
        self.parseTimer.setSingleShot(False)
        self.parseTimer.timeout.connect(self.parseData)

        try:
            if os.name == "nt":
                uart = "COM" + cliCom.text()
                data = "COM" + dataCom.text()
            else:
                uart = cliCom.text()
                data = dataCom.text()

            log.info(f"Menghubungkan ke port COM: CLI={uart}, DATA={data}")
            self.parser.connectComPorts(uart, data)
            connectStatus.setText("Connected")
            log.info("Berhasil terhubung ke port COM")
        except Exception as e:
            log.error(f"Gagal terhubung ke port COM: {e}")
            connectStatus.setText("Unable to Connect")
            return -1

        # Inisialisasi thread klasifikasi jika belum diinisialisasi
        if self.classification_thread is None:
            try:
                # Muat konfigurasi klasifikasi
                log.info("Mencoba memuat konfigurasi klasifikasi dari config_klasifikasi.json")
                with open('config_klasifikasi.json', 'r') as f:
                    config_klasifikasi = json.load(f)
                model_path = config_klasifikasi.get("model_path", "D:\Alif\Kuliah\TA\Realtime GUI V3\common\ModelPointCNN.keras")
                scaler_path = config_klasifikasi.get("scaler_path", "D:\Alif\Kuliah\TA\Realtime GUI V3\common\Scaler.joblib")
                log.info(f"Konfigurasi klasifikasi berhasil dimuat: model_path={model_path}, scaler_path={scaler_path}")
                self.initClassificationThread(model_path, scaler_path)
            except Exception as e:
                log.error(f"Error saat memuat konfigurasi klasifikasi: {e}")
                log.info("Menggunakan path default untuk model dan scaler")
                # Gunakan path default jika file konfigurasi tidak ditemukan
                self.initClassificationThread("D:\Alif\Kuliah\TA\Realtime GUI V3\common\ModelPointCNN.keras", "D:\Alif\Kuliah\TA\Realtime GUI V3\common\Scaler.joblib")

        return 0

    def startApp(self):
        if (self.replay and self.playing is False):
            log.info("Memulai replay data")
            self.replayTimer = QTimer()
            self.replayTimer.setSingleShot(True)
            self.replayTimer.timeout.connect(self.replayData)
            self.playing = True
            self.replayTimer.start(100) # arbitrary value to start plotting
        elif (self.replay and self.playing is True):
            log.info("Menjeda replay data")
            self.playing = False
        else:
            log.info(f"Memulai aplikasi dengan interval parsing {self.frameTime} ms")
            self.parseTimer.start(int(self.frameTime)) # need this line, this is for normal plotting

    def loadForReplay(self, state):
        if (state):
            log.info("Memuat data untuk replay")
            self.cachedData.setCachedRecord = "True"
            with open(self.replayFile[0], 'r') as fp:
                self.data = json.load(fp)
            log.info(f"File replay berhasil dimuat: {len(self.data['data'])} frame")
            self.parseCfg("")
            self.sl.setMinimum(0)
            self.sl.setMaximum(len(self.data['data']) - 1)
            self.sl.setValue(0)
            self.sl.setTickInterval(5)
        else:
            log.info("Menghentikan mode replay")
            self.cachedData.setCachedRecord = "False"

    def replayData(self):
        if (self.playing):
            outputDict = self.data['data'][self.replayFrameNum]['frameData']
            log.debug(f"Replay frame {self.replayFrameNum}")
            self.updateGraph(outputDict)
            self.replayFrameNum += 1
            self.sl.setValue(self.replayFrameNum)
            if (self.replayFrameNum < len(self.data['data'])):
                next_time = self.data['data'][self.replayFrameNum]['timestamp'] - self.data['data'][self.replayFrameNum-1]['timestamp']
                self.replayTimer.start(next_time)
                log.debug(f"Next frame dalam {next_time} ms")

    def sliderValueChange(self):
        self.replayFrameNum = self.sl.value()
        log.debug(f"Slider diubah ke frame {self.replayFrameNum}")

    def parseData(self):
        self.uart_thread.start(priority=QThread.HighestPriority)

    def gracefulReset(self):
        log.info("Melakukan reset koneksi")
        self.parseTimer.stop()
        self.uart_thread.stop()
        
        # Hentikan thread klasifikasi jika ada
        if hasattr(self, 'classification_thread') and self.classification_thread is not None:
            log.info("Menghentikan thread klasifikasi")
            self.classification_thread.stop()
            self.classification_thread = None
            self.classification_result = None

        if self.parser.cliCom is not None:
            log.info("Menutup port CLI COM")
            self.parser.cliCom.close()
        if self.parser.dataCom is not None:
            log.info("Menutup port DATA COM")
            self.parser.dataCom.close()

        for demo in self.demoClassDict.values():
            if hasattr(demo, "plot_3d_thread"):
                log.info("Menghentikan thread plot 3D")
                demo.plot_3d_thread.stop()
            if hasattr(demo, "plot_3d"):
                log.info("Menghapus semua bounding box")
                demo.removeAllBoundBoxes()
        
        log.info("Reset koneksi selesai")
