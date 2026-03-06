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
            <el-select v-model="selectedModel" class="model-select" :placeholder="$t('analyze.modelPlaceholder')">
              <el-option label="YOLO26（默认推荐）" value="yolo26" />
              <el-option label="YOLOv8" value="yolov8" />
            </el-select>
          </el-form-item>
        </el-form>

        <div class="upload-area">
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
              算法输出图像 (包含检测框)
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
            <span class="value highlight">{{ result.plateNumber }}</span>
          </div>
          <div class="result-item">
            <span class="label">车牌属性</span>
            <span class="value">{{ result.plateType || '-' }}</span>
          </div>
          <div class="result-item">
            <span class="label">{{ $t('analyze.modelUsed') }}</span>
            <span class="value">
              <el-tag :type="result.modelType === 'yolov8' ? 'warning' : 'success'" size="small">
                {{ result.modelType === 'yolov8' ? 'YOLOv8' : 'YOLO26' }}
              </el-tag>
            </span>
          </div>
          <div class="result-item">
            <span class="label">{{ $t('analyze.timeTaken') }}</span>
            <span class="value">{{ result.processingTimeMs }}ms</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { UploadFilled, Picture } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const { t } = useI18n()

const selectedModel = ref('yolo26')
const previewUrl = ref('')
const selectedFile = ref(null)
const analyzing = ref(false)
const result = ref(null)

const handleFileChange = (file) => {
  if (file.raw.type.startsWith('image/')) {
    selectedFile.value = file.raw
    previewUrl.value = URL.createObjectURL(file.raw)
    result.value = null // Reset previous result
  } else {
    ElMessage.error('Please upload a valid image file.')
  }
}

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
      ElMessage.success('识别完成')
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
      ElMessage.error(data.message || '识别失败')
    }
  } catch (err) {
    ElMessage.error('网络请求失败，请检查后端是否启动')
    console.error(err)
  } finally {
    analyzing.value = false
  }
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
  margin-top: auto;
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
</style>
