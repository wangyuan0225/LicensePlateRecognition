<template>
  <div class="analyze-container page-container">
    <div class="header-section">
      <h1 class="page-title">{{ $t('analyze.title') }}</h1>
      <p class="page-subtitle">{{ $t('analyze.subtitle') }}</p>
    </div>

    <div class="content-grid">
      <!-- Upload and Control Section -->
      <div class="control-panel notion-card">
        <h3 class="panel-title">{{ $t('analyze.configPanel') }}</h3>

        <el-form label-position="top">
          <el-form-item :label="$t('analyze.modelLabel')">
            <el-select v-model="selectedModel" :placeholder="$t('analyze.modelPlaceholder')" class="model-select">
              <el-option :label="$t('analyze.modelYoloFast')" value="yolo26" />
              <el-option :label="$t('analyze.modelYoloAcc')" value="yolov8" />
              <el-option :label="$t('analyze.modelHyperLPR')" value="hyperlpr" />
              <el-option :label="$t('analyze.modelFusion')" value="fusion" />
            </el-select>
          </el-form-item>
        </el-form>

        <!-- Input Mode Toggle -->
        <div class="input-mode-toggle">
          <el-radio-group v-model="inputMode" size="default" @change="onInputModeChange">
            <el-radio-button value="upload">
              <el-icon style="margin-right:4px">
                <UploadFilled />
              </el-icon>{{ $t('analyze.inputModeUpload') }}
            </el-radio-button>
            <el-radio-button value="camera">
              <el-icon style="margin-right:4px">
                <VideoCameraFilled />
              </el-icon>{{ $t('analyze.inputModeCamera') }}
            </el-radio-button>
          </el-radio-group>
        </div>

        <!-- Upload Mode -->
        <div v-if="inputMode === 'upload'" class="upload-area">
          <el-upload class="plate-uploader" drag action="#" :auto-upload="false" :show-file-list="false"
            :on-change="handleFileChange" accept="image/*">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              {{ $t('analyze.dropText') }} <em>{{ $t('analyze.clickUpload') }}</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                {{ $t('analyze.uploadTip') }}
              </div>
            </template>
          </el-upload>
        </div>

        <!-- Camera Mode -->
        <div v-if="inputMode === 'camera'" class="camera-area">
          <div class="camera-preview-wrapper">
            <video v-show="cameraActive && !capturedPhoto" ref="videoRef" autoplay playsinline muted
              class="camera-video" />
            <img v-if="capturedPhoto" :src="capturedPhoto" class="camera-snapshot" alt="Captured" />
            <div v-if="!cameraActive && !capturedPhoto" class="camera-placeholder">
              <el-icon :size="36" color="#aaa">
                <VideoCameraFilled />
              </el-icon>
              <p>{{ $t('analyze.cameraPlaceholder') }}</p>
            </div>
          </div>

          <div class="camera-controls">
            <el-button v-if="!cameraActive && !capturedPhoto" type="primary" @click="startCamera"
              :icon="VideoCameraFilled">
              {{ $t('analyze.btnStartCamera') }}
            </el-button>
            <div v-if="cameraActive && !capturedPhoto" class="capture-btn" @click="capturePhoto">
              <span class="capture-inner"></span>
            </div>
            <div v-if="cameraActive && !capturedPhoto" class="close-camera-btn" @click="stopCamera">
              <span class="close-x">✕</span>
            </div>
            <el-button v-if="capturedPhoto" type="warning" @click="retakePhoto" plain>
              {{ $t('analyze.btnRetake') }}
            </el-button>
          </div>
        </div>

        <el-button type="primary" class="analyze-btn" :disabled="!previewUrl" :loading="analyzing"
          @click="startAnalysis">
          {{ analyzing ? $t('analyze.processing') : $t('analyze.runBtn') }}
        </el-button>
      </div>

      <!-- Preview and Results Section -->
      <div class="results-panel notion-card">
        <h3 class="panel-title">{{ $t('analyze.viewerPanel') }}</h3>

        <div class="image-viewer" :class="{ 'has-image': previewUrl }">
          <template v-if="previewUrl">
            <img :src="result ? result.resultImageUrl : previewUrl" class="preview-image" alt="Preview" />

            <div v-if="result && !analyzing" class="result-badge">
              {{ $t('analyze.resultBadge') }}
            </div>
          </template>
          <template v-else>
            <div class="empty-state">
              <el-icon :size="48" color="var(--border-color)">
                <Picture />
              </el-icon>
              <p>{{ $t('analyze.noImage') }}</p>
            </div>
          </template>
        </div>

        <div class="results-data" v-if="result && !analyzing">
          <div class="result-item">
            <span class="label">{{ $t('analyze.plateNumber') }}</span>
            <span class="value custom-tag" :style="getPlateColorStyle(result.plateType)">{{ result.plateNumber }}</span>
          </div>
          <div class="result-item">
            <span class="label">{{ $t('analyze.plateType') }}</span>
            <span class="value custom-tag" :style="getPlateColorStyle(result.plateType)">{{ result.plateType || '-'
            }}</span>
          </div>
          <div class="result-item">
            <span class="label">{{ $t('analyze.modelUsed') }}</span>
            <span class="value">
              <span class="custom-tag">
                {{ $t(result.modelType === 'yolov8' ? 'analyze.modelNameYolov8' : (result.modelType === 'hyperlpr' ?
                  'analyze.modelNameHyperLPR' : (result.modelType === 'fusion' ? 'analyze.modelNameFusion' :
                    'analyze.modelNameYolo26'))) }}
              </span>
            </span>
          </div>
          <div class="result-item">
            <span class="label">{{ $t('analyze.timeTaken') }}</span>
            <span class="value">{{ result.processingTimeMs }}ms</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Hidden canvas for capturing camera frame -->
    <canvas ref="canvasRef" style="display:none" />
  </div>
</template>

<script setup>
import { ref, onBeforeUnmount } from 'vue'
import { useI18n } from 'vue-i18n'
import { UploadFilled, Picture, VideoCameraFilled } from '@element-plus/icons-vue'
import { useMessage } from '@/composables/useMessage'

const { t } = useI18n()
const message = useMessage()

const selectedModel = ref('yolo26')
const previewUrl = ref('')
const selectedFile = ref(null)
const analyzing = ref(false)
const result = ref(null)

// --- Input mode ---
const inputMode = ref('upload')

// --- Camera state ---
const cameraActive = ref(false)
const capturedPhoto = ref('')
const videoRef = ref(null)
const canvasRef = ref(null)
let mediaStream = null

const onInputModeChange = () => {
  // Cleanup when switching modes
  stopCamera()
  capturedPhoto.value = ''
  result.value = null
  if (inputMode.value === 'upload') {
    // Keep previewUrl if file was selected
  } else {
    previewUrl.value = ''
    selectedFile.value = null
  }
}

// --- Upload mode ---
const handleFileChange = (file) => {
  if (file.raw.type.startsWith('image/')) {
    selectedFile.value = file.raw
    previewUrl.value = URL.createObjectURL(file.raw)
    result.value = null
  } else {
    message.error('Please upload a valid image file.')
  }
}

// --- Camera mode ---
const startCamera = async () => {
  try {
    // Prefer rear camera for phone, any camera for desktop
    const constraints = {
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      }
    }
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
    if (videoRef.value) {
      videoRef.value.srcObject = mediaStream
    }
    cameraActive.value = true
    capturedPhoto.value = ''
    previewUrl.value = ''
    result.value = null
  } catch (err) {
    if (err.name === 'NotAllowedError') {
      message.error('相机权限被拒绝，请在浏览器设置中允许使用相机')
    } else if (err.name === 'NotFoundError') {
      message.error('未检测到摄像头设备')
    } else {
      message.error('无法打开相机: ' + err.message)
    }
  }
}

const stopCamera = () => {
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop())
    mediaStream = null
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null
  }
  cameraActive.value = false
}

const capturePhoto = () => {
  const video = videoRef.value
  const canvas = canvasRef.value
  if (!video || !canvas) return

  // To match CSS object-fit: cover, we need to calculate the actual visible area
  const videoRatio = video.videoWidth / video.videoHeight
  const elementRatio = video.clientWidth / video.clientHeight

  let sourceX = 0
  let sourceY = 0
  let sourceWidth = video.videoWidth
  let sourceHeight = video.videoHeight

  if (videoRatio > elementRatio) {
    // Video is wider than element -> crop sides
    sourceWidth = video.videoHeight * elementRatio
    sourceX = (video.videoWidth - sourceWidth) / 2
  } else {
    // Video is taller than element -> crop top/bottom
    sourceHeight = video.videoWidth / elementRatio
    sourceY = (video.videoHeight - sourceHeight) / 2
  }

  // Set canvas size to the cropped resolution (high quality)
  canvas.width = sourceWidth
  canvas.height = sourceHeight
  const ctx = canvas.getContext('2d')

  // Draw only the cropped portion
  ctx.drawImage(
    video,
    sourceX, sourceY, sourceWidth, sourceHeight, // Source Rect
    0, 0, sourceWidth, sourceHeight              // Destination Rect
  )

  // Convert to blob and create preview
  canvas.toBlob((blob) => {
    if (blob) {
      const file = new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' })
      selectedFile.value = file
      capturedPhoto.value = URL.createObjectURL(blob)
      previewUrl.value = capturedPhoto.value
      result.value = null
      stopCamera()
    }
  }, 'image/jpeg', 0.92)
}

const retakePhoto = () => {
  capturedPhoto.value = ''
  previewUrl.value = ''
  selectedFile.value = null
  result.value = null
  startCamera()
}

// Cleanup on component unmount
onBeforeUnmount(() => {
  stopCamera()
})

// --- Analysis (shared by both modes) ---
const startAnalysis = async () => {
  if (!selectedFile.value) return

  analyzing.value = true
  result.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    formData.append('modelType', selectedModel.value)

    const token = localStorage.getItem('token')
    const headers = {}
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    const res = await fetch('/api/v1/analyze/upload', {
      method: 'POST',
      headers,
      body: formData,
    })

    const data = await res.json()

    if (data.code === 200) {
      message.success('识别完成')
      result.value = {
        plateNumber: data.data.plateNumber,
        plateType: data.data.plateType,
        confidence: data.data.confidence,
        processingTimeMs: data.data.processingTimeMs,
        modelType: data.data.modelType,
        resultImageUrl: data.data.resultImageUrl,
        originalImageUrl: data.data.originalImageUrl,
      }
    } else {
      message.error(data.message || '识别失败')
    }
  } catch (err) {
    message.error('网络请求失败，请检查后端是否启动')
    console.error(err)
  } finally {
    analyzing.value = false
  }
}

const getPlateColorStyle = (text) => {
  if (!text) return { color: '#000' }
  // 车牌类型包含“牌”或“车”，一律返回黑色
  if (text.includes('牌') || text.includes('车')) return { color: '#000' }

  if (text.includes('蓝')) return { color: '#0050b3' } // 蓝色
  if (text.includes('黄')) return { color: '#d4b106' } // 黄色
  if (text.includes('绿')) return { color: '#389e0d' } // 绿色
  if (text.includes('白')) return { color: '#595959' } // 白色
  if (text.includes('黑')) return { color: '#000000' } // 黑色
  return { color: '#000' }
}
</script>

<style scoped>
.analyze-container {
  max-width: 1200px;
}

.header-section {
  margin-bottom: 32px;
}

.page-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--text-primary);
  letter-spacing: -0.02em;
}

.page-subtitle {
  color: var(--text-secondary);
  font-size: 1rem;
}

.content-grid {
  display: grid;
  grid-template-columns: 380px 1fr;
  align-items: start;
  gap: 24px;
}

.panel-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 24px;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 16px;
}

.control-panel {
  display: flex;
  flex-direction: column;
}

.model-select {
  width: 100%;
}

.upload-area {
  margin-top: 16px;
  margin-bottom: 24px;
}

:deep(.el-upload-dragger) {
  border: 1px dashed var(--border-color);
  border-radius: 8px;
  background-color: var(--bg-secondary);
  transition: all 0.2s;
}

:deep(.el-upload-dragger:hover) {
  border-color: var(--primary-color);
  background-color: white;
}

:deep(.el-icon--upload) {
  color: var(--text-secondary) !important;
}

.el-upload__tip {
  color: var(--text-secondary);
  text-align: center;
  margin-top: 8px;
}

.analyze-btn {
  width: 100%;
  height: 48px;
  font-size: 1.05rem;
}

.results-panel {
  display: flex;
  flex-direction: column;
}

.image-viewer {
  flex: 1;
  min-height: 400px;
  background: var(--bg-secondary);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
  border: 1px solid var(--border-color);
  margin-bottom: 24px;
}

.image-viewer.has-image {
  background: #000;
}

.preview-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  color: var(--text-secondary);
}

.result-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  background: var(--success-color);
  color: white;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  opacity: 0.9;
}

.results-data {
  background: var(--bg-secondary);
  border-radius: 8px;
  padding: 20px;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  border: 1px solid var(--border-color);
}

.result-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.result-item .label {
  font-size: 0.85rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
}

.result-item .value {
  font-size: 1.1rem;
  font-weight: 500;
  color: var(--text-primary);
}

.result-item .value.highlight {
  color: var(--success-color);
  font-weight: 700;
  font-size: 1.25rem;
}

@media (max-width: 900px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}

/* --- Input mode toggle --- */
.input-mode-toggle {
  margin-bottom: 16px;
  display: flex;
  justify-content: center;
}

.input-mode-toggle :deep(.el-radio-button__inner) {
  display: flex;
  align-items: center;
}

/* --- Camera mode --- */
.camera-area {
  margin-top: 8px;
  margin-bottom: 24px;
}

.camera-preview-wrapper {
  width: 100%;
  height: 240px;
  background: #1a1a2e;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-color);
  margin-bottom: 12px;
}

.camera-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.camera-snapshot {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.camera-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: #888;
}

.camera-placeholder p {
  font-size: 0.85rem;
}

.camera-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
}

.capture-btn {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  border: 3px solid #333;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.15s;
  background: transparent;
}

.capture-btn:hover {
  transform: scale(1.08);
}

.capture-btn:active {
  transform: scale(0.95);
}

.capture-inner {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #1a1a2e;
}

.close-camera-btn {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: #e53935;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.15s, background 0.2s;
}

.close-camera-btn:hover {
  background: #c62828;
  transform: scale(1.08);
}

.close-camera-btn:active {
  transform: scale(0.95);
}

.close-x {
  color: #fff;
  font-size: 22px;
  font-weight: 700;
  line-height: 1;
}
</style>
