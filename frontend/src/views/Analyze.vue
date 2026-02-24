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
              <el-option :label="$t('analyze.modelYoloFast')" value="yolov8_fast" />
              <el-option :label="$t('analyze.modelYoloAcc')" value="yolov8_acc" />
              <el-option :label="$t('analyze.modelResNet')" value="resnet50" />
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
            <img :src="previewUrl" class="preview-image" alt="Preview" />

            <!-- Simulated bounding box for demonstration -->
            <div v-if="result && !analyzing" class="bounding-box-overlay">
              <div class="box">
                <span class="label">{{ result.plateNumber }} ({{ (result.confidence * 100).toFixed(1) }}%)</span>
              </div>
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
            <span class="label">{{ $t('analyze.confidence') }}</span>
            <span class="value">{{ (result.confidence * 100).toFixed(2) }}%</span>
          </div>
          <div class="result-item">
            <span class="label">{{ $t('analyze.modelUsed') }}</span>
            <span class="value">{{ getModelName(result.model) }}</span>
          </div>
          <div class="result-item">
            <span class="label">{{ $t('analyze.timeTaken') }}</span>
            <span class="value">{{ result.processingTime }}ms</span>
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

const selectedModel = ref('yolov8_fast')
const previewUrl = ref('')
const selectedFile = ref(null)
const analyzing = ref(false)
const result = ref(null)

const getModelName = (val) => {
  const map = {
    'yolov8_fast': t('analyze.modelYoloFast'),
    'yolov8_acc': t('analyze.modelYoloAcc'),
    'resnet50': t('analyze.modelResNet')
  }
  return map[val] || val
}

const handleFileChange = (file) => {
  if (file.raw.type.startsWith('image/')) {
    selectedFile.value = file.raw
    previewUrl.value = URL.createObjectURL(file.raw)
    result.value = null // Reset previous result
  } else {
    ElMessage.error('Please upload a valid image file.')
  }
}

const startAnalysis = () => {
  if (!selectedFile.value) return

  analyzing.value = true
  result.value = null

  // Simulate API call processing
  setTimeout(() => {
    analyzing.value = false
    ElMessage.success('Analysis completed successfully.')

    // Mock result data
    result.value = {
      plateNumber: '沪A·88888',
      confidence: 0.985,
      model: selectedModel.value,
      processingTime: 124,
      // Bounding box coords could go here in a real app
    }
  }, 1500)
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

.bounding-box-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Mock positioning for the demo */
.bounding-box-overlay .box {
  width: 30%;
  height: 20%;
  border: 2px solid var(--success-color);
  background: rgba(15, 123, 108, 0.1);
  position: relative;
  border-radius: 2px;
}

.bounding-box-overlay .label {
  position: absolute;
  top: -28px;
  left: -2px;
  background: var(--success-color);
  color: white;
  padding: 4px 8px;
  font-size: 0.85rem;
  font-weight: 600;
  border-radius: 4px;
  white-space: nowrap;
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
