<template>
  <div class="feedback-container page-container">
    <div class="header-section">
      <h1 class="page-title">{{ $t('feedback.title') }}</h1>
      <p class="page-subtitle">{{ $t('feedback.subtitle') }}</p>
    </div>

    <div class="notion-card table-card" v-loading="loading">
      <el-table :data="tableData" style="width: 100%" :empty-text="$t('feedback.noData')">
        <el-table-column prop="createdAt" :label="$t('feedback.colTime')" width="180">
          <template #default="scope">
            <span>{{ formatDate(scope.row.createdAt) }}</span>
          </template>
        </el-table-column>
        <el-table-column :label="$t('feedback.colOriginal')" width="120" align="center">
          <template #default="scope">
            <el-image
              v-if="scope.row.originalImageUrl"
              style="width: 80px; height: 60px; border-radius: 4px;"
              :src="scope.row.originalImageUrl"
              :preview-src-list="[scope.row.originalImageUrl]"
              fit="cover"
              preview-teleported
            />
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column :label="$t('feedback.colResult')" width="120" align="center">
          <template #default="scope">
            <el-image
              v-if="scope.row.resultImageUrl"
              style="width: 80px; height: 60px; border-radius: 4px;"
              :src="scope.row.resultImageUrl"
              :preview-src-list="[scope.row.resultImageUrl]"
              fit="cover"
              preview-teleported
            />
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="recognizedPlate" :label="$t('feedback.colRecognized')" width="140">
          <template #default="scope">
            <span class="custom-tag">{{ scope.row.recognizedPlate || '-' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="correctedPlate" :label="$t('feedback.colCorrected')" width="140">
          <template #default="scope">
            <span class="custom-tag highlight-tag">{{ scope.row.correctedPlate || '-' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="modelType" :label="$t('feedback.colModel')" min-width="120">
          <template #default="scope">
            <span class="custom-tag model-tag">{{ getModelName(scope.row.modelType) }}</span>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useMessage } from '@/composables/useMessage'
import { useRouter } from 'vue-router'

const { t } = useI18n()
const message = useMessage()
const router = useRouter()

const loading = ref(false)
const tableData = ref([])

const fetchFeedbacks = async () => {
  loading.value = true
  try {
    const token = localStorage.getItem('token')
    if (!token) {
      router.push('/login')
      return
    }

    const res = await fetch('/api/v1/feedback/list', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
    const data = await res.json()
    if (data.code === 200) {
      tableData.value = data.data
    } else {
      message.error(data.message || t('feedback.fetchFail'))
    }
  } catch (err) {
    message.error(t('feedback.fetchFail'))
    console.error(err)
  } finally {
    loading.value = false
  }
}

const formatDate = (dateString) => {
  if (!dateString) return '-'
  try {
    const date = new Date(dateString)
    return date.toLocaleString()
  } catch {
    return dateString
  }
}

const getModelName = (modelType) => {
  if (!modelType) return '-'
  if (modelType === 'yolov8') return t('analyze.modelNameYolov8')
  if (modelType === 'hyperlpr') return t('analyze.modelNameHyperLPR')
  if (modelType === 'fusion') return t('analyze.modelNameFusion')
  return t('analyze.modelNameYolo26')
}

onMounted(() => {
  fetchFeedbacks()
})
</script>

<style scoped>
.feedback-container {
  max-width: 1200px;
}

.header-section {
  margin-bottom: 24px;
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

.table-card {
  padding: 24px;
  overflow: hidden;
}

.highlight-tag {
  color: var(--primary-color) !important;
  font-weight: 600 !important;
  background-color: var(--bg-color) !important;
  border-color: var(--primary-color) !important;
}

.model-tag {
  background-color: #f0f5ff;
  color: #2f54eb;
  border-color: #adc6ff;
}
</style>
