<template>
  <div class="feedback-container page-container">
    <div class="header-section" style="display: flex; justify-content: space-between; align-items: flex-end;">
      <div class="title-area">
        <h1 class="page-title">{{ $t('feedback.title') }}</h1>
        <p class="page-subtitle">{{ $t('feedback.subtitle') }}</p>
      </div>
      <div class="actions-area">
        <el-button v-if="selectedRows.length > 0" class="notion-btn danger-btn" @click="batchRevokeFeedback">
          批量撤回
        </el-button>
      </div>
    </div>

    <div class="notion-card table-wrapper" v-loading="loading">
      <el-table :data="tableData" style="width: 100%" @selection-change="handleSelectionChange"
        :empty-text="$t('feedback.noData')">
        <el-table-column type="selection" width="55" align="center" />
        <el-table-column prop="createdAt" :label="$t('feedback.colTime')" width="180">
          <template #default="scope">
            <span>{{ formatDate(scope.row.createdAt) }}</span>
          </template>
        </el-table-column>
        <el-table-column :label="$t('feedback.colOriginal')" width="160" align="center">
          <template #default="scope">
            <el-image v-if="scope.row.originalImageUrl" style="width: 120px; height: 90px; border-radius: 4px;"
              :src="scope.row.originalImageUrl" :preview-src-list="[scope.row.originalImageUrl]" fit="cover"
              preview-teleported />
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column :label="$t('feedback.colResult')" width="160" align="center">
          <template #default="scope">
            <el-image v-if="scope.row.resultImageUrl" style="width: 120px; height: 90px; border-radius: 4px;"
              :src="scope.row.resultImageUrl" :preview-src-list="[scope.row.resultImageUrl]" fit="cover"
              preview-teleported />
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="recognizedPlate" :label="$t('feedback.colRecognized')" width="140" align="center">
          <template #default="scope">
            <span class="custom-tag" :style="getPlateColorStyle(scope.row.recognizedPlate)">{{ scope.row.recognizedPlate
              || '-' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="correctedPlate" :label="$t('feedback.colCorrected')" width="140" align="center">
          <template #default="scope">
            <span class="custom-tag" :style="getPlateColorStyle(scope.row.correctedPlate)">{{ scope.row.correctedPlate
              || '-' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="modelType" :label="$t('feedback.colModel')" min-width="100" align="center">
          <template #default="scope">
            <span class="custom-tag">{{ getModelName(scope.row.modelType) }}</span>
          </template>
        </el-table-column>
        <el-table-column :label="$t('feedback.colStatus')" min-width="100" align="center">
          <template #default="scope">
            <span class="custom-tag" :style="getStatusColorStyle(scope.row.status)">
              {{ getStatusText(scope.row.status) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column :label="$t('history.colActions')" width="120" align="center" fixed="right">
          <template #default="scope">
            <el-button class="notion-btn danger-btn" size="small" @click="revokeFeedback(scope.row)">
              {{ $t('app.actionRevoke') }}
            </el-button>
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
const selectedRows = ref([])

const handleSelectionChange = (val) => {
  selectedRows.value = val
}

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
  if (modelType === 'yolov11') return t('analyze.modelNameYolov11')
  return t('analyze.modelNameYolo26')
}

const getStatusColorStyle = (status) => {
  switch (status) {
    case 'APPROVED': return { color: '#16a34a' }
    case 'REJECTED': return { color: '#dc2626' }
    default: return { color: '#d97706' }
  }
}

const getStatusText = (status) => {
  switch (status) {
    case 'APPROVED': return t('app.statusApproved')
    case 'REJECTED': return t('app.statusRejected')
    default: return t('app.statusPending')
  }
}

const getPlateColorStyle = (text) => {
  if (!text) return { color: '#000' }
  if (text.includes('牌') || text.includes('车')) return { color: '#000' }

  if (text.includes('蓝')) return { color: '#0050b3' }
  if (text.includes('黄')) return { color: '#d4b106' }
  if (text.includes('绿')) return { color: '#389e0d' }
  if (text.includes('白')) return { color: '#595959' }
  if (text.includes('黑')) return { color: '#000000' }
  return { color: '#000' }
}

import { ElMessageBox } from 'element-plus'

const batchRevokeFeedback = () => {
  ElMessageBox.confirm(
    '是否确认撤回选中的记录？',
    t('history.confirmTitle'),
    {
      confirmButtonText: t('history.confirmOk'),
      cancelButtonText: t('history.confirmCancel'),
      type: 'warning',
      customClass: 'notion-msgbox'
    }
  ).then(async () => {
    try {
      const token = localStorage.getItem('token')
      const promises = selectedRows.value.map(row =>
        fetch(`/api/v1/feedback/${row.id}`, {
          method: 'DELETE',
          headers: { 'Authorization': `Bearer ${token}` }
        }).then(res => res.json())
      )
      await Promise.all(promises)
      message.success('批量撤回完成')
      fetchFeedbacks()
    } catch (err) {
      console.error(err)
      message.error('批量撤回出错')
    }
  }).catch(() => { })
}

const revokeFeedback = (row) => {
  ElMessageBox.confirm(
    t('history.confirmDelete'),
    t('history.confirmTitle'),
    {
      confirmButtonText: t('history.confirmOk'),
      cancelButtonText: t('history.confirmCancel'),
      type: 'warning',
      customClass: 'notion-msgbox'
    }
  ).then(async () => {
    try {
      const token = localStorage.getItem('token')
      const res = await fetch(`/api/v1/feedback/${row.id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      const data = await res.json()
      if (data.code === 200) {
        message.success('撤回成功')
        fetchFeedbacks()
      } else {
        message.error(data.message || '撤回失败')
      }
    } catch (err) {
      console.error(err)
      message.error('撤回失败')
    }
  }).catch(() => { })
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

.table-wrapper {
  padding: 0;
  overflow: hidden;
}

:deep(.el-table) {
  --el-table-border-color: var(--border-color);
  --el-table-header-bg-color: var(--bg-secondary);
  --el-table-header-text-color: var(--text-primary);
  --el-table-text-color: var(--text-primary);
  --el-table-row-hover-bg-color: var(--bg-secondary);
}

:deep(.el-table th.el-table__cell) {
  font-weight: 600;
  background-color: var(--bg-secondary) !important;
}

.custom-tag {
  display: inline-block;
  border: 1px solid #000;
  background-color: #fff;
  color: #000;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.5;
}

.notion-btn {
  font-weight: 600 !important;
  border: 1px solid rgba(0, 0, 0, 0.1) !important;
  color: #fff !important;
  border-radius: 6px !important;
  transition: all 0.2s ease;
  margin: 0 !important;
}

.danger-btn {
  background-color: #dc2626 !important;
}

.danger-btn:hover {
  background-color: #b91c1c !important;
}
</style>
