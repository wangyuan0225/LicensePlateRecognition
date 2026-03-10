<template>
  <div class="history-container page-container">
    <div class="header-section">
      <div class="title-area">
        <h1 class="page-title">{{ $t('history.title') }}</h1>
        <p class="page-subtitle">{{ $t('history.subtitle') }}</p>
      </div>
      <div class="actions-area">
        <el-button v-if="selectedRows.length > 0" class="notion-btn danger-btn"
          @click="batchDeleteRecords">批量删除</el-button>
        <el-input v-model="searchQuery" :placeholder="$t('history.searchPlaceholder')" prefix-icon="Search"
          class="search-input" clearable @clear="fetchData" @keyup.enter="fetchData" />
        <el-date-picker v-model="dateRange" type="daterange" range-separator="-"
          :start-placeholder="$t('history.startDate')" :end-placeholder="$t('history.endDate')" class="date-picker"
          @change="fetchData" />
        <el-button type="primary" icon="Search" @click="fetchData">{{ $t('history.searchBtn') }}</el-button>
      </div>
    </div>

    <div class="notion-card table-wrapper">
      <el-table :data="tableData" style="width: 100%" v-loading="loading" :row-class-name="tableRowClassName"
        @selection-change="handleSelectionChange">
        <el-table-column type="selection" width="55" align="center" />
        <el-table-column prop="createdAt" :label="$t('history.colDate')" min-width="220" align="center">
          <template #default="scope">
            <span class="date-cell">
              <el-icon>
                <Calendar />
              </el-icon>
              {{ scope.row.createdAt }}
            </span>
          </template>
        </el-table-column>

        <el-table-column prop="plateNumber" :label="$t('history.colPlate')" min-width="180" align="center">
          <template #default="scope">
            <span class="custom-tag" :style="getPlateColorStyle(scope.row.plateType)">
              {{ scope.row.plateNumber || '-' }}
            </span>
          </template>
        </el-table-column>

        <el-table-column prop="plateType" :label="$t('history.colPlateType')" min-width="120" align="center">
          <template #default="scope">
            <span class="custom-tag" :style="getPlateColorStyle(scope.row.plateType)">
              {{ scope.row.plateType || '-' }}
            </span>
          </template>
        </el-table-column>

        <el-table-column prop="modelType" :label="$t('history.colModel')" min-width="140" align="center">
          <template #default="scope">
            <span class="custom-tag">
              {{ $t(scope.row.modelType === 'yolov8' ? 'analyze.modelNameYolov8' : (scope.row.modelType === 'hyperlpr' ?
                'analyze.modelNameHyperLPR' : (scope.row.modelType === 'fusion' ? 'analyze.modelNameFusion' :
                  (scope.row.modelType === 'yolov11' ? 'analyze.modelNameYolov11' : 'analyze.modelNameYolo26')))) }}
            </span>
          </template>
        </el-table-column>

        <el-table-column prop="processingTimeMs" :label="$t('history.colTime')" min-width="120">
          <template #default="scope">
            <span class="time-cell">{{ scope.row.processingTimeMs ? scope.row.processingTimeMs.toFixed(1) : '-'
            }}ms</span>
          </template>
        </el-table-column>

        <el-table-column :label="$t('history.colThumb')" width="140" align="center">
          <template #default="scope">
            <el-image :src="scope.row.thumbnailUrl" class="thumb-img"
              :preview-src-list="[scope.row.resultImageUrl, scope.row.originalImageUrl]" preview-teleported
              fit="cover" />
          </template>
        </el-table-column>

        <el-table-column fixed="right" :label="$t('history.colActions')" width="240" align="center">
          <template #default="scope">
            <div class="action-btn-group">
              <el-button class="notion-btn primary-btn" size="small" @click="showDetail(scope.row)">
                {{ $t('history.actionDetails') }}
              </el-button>
              <el-button class="notion-btn warning-btn" size="small" @click="openFeedback(scope.row)">
                {{ $t('feedback.btnFeedbackError') }}
              </el-button>
              <el-button class="notion-btn danger-btn" size="small" @click="deleteRecord(scope.row)">
                {{ $t('history.actionDelete') }}
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-wrapper">
        <el-pagination v-model:current-page="currentPage" v-model:page-size="pageSize" :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper" :total="total" @current-change="fetchData"
          @size-change="fetchData" />
      </div>
    </div>

    <!-- Detail Dialog -->
    <el-dialog v-model="detailVisible" :title="$t('history.dialogTitle')" width="800px">
      <div v-if="detailRecord" class="detail-content">
        <div class="detail-images">
          <div class="detail-image-box">
            <h4>{{ $t('history.detailOriginal') }}</h4>
            <el-image :src="detailRecord.originalImageUrl" fit="contain" style="width: 100%; height: 300px;" />
          </div>
          <div class="detail-image-box">
            <h4>{{ $t('history.detailResult') }}</h4>
            <el-image :src="detailRecord.resultImageUrl" fit="contain" style="width: 100%; height: 300px;" />
          </div>
        </div>
        <div class="detail-info">
          <el-descriptions :column="2" border>
            <el-descriptions-item :label="$t('history.detailPlate')">
              <span class="custom-tag" :style="getPlateColorStyle(detailRecord.plateType)">{{ detailRecord.plateNumber
              }}</span>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('history.detailType')">
              <span class="custom-tag" :style="getPlateColorStyle(detailRecord.plateType)">{{ detailRecord.plateType ||
                '-'
              }}</span>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('history.detailModel')">
              <span class="custom-tag">
                {{ $t(detailRecord.modelType === 'yolov8' ? 'analyze.modelNameYolov8' : (detailRecord.modelType ===
                  'hyperlpr' ? 'analyze.modelNameHyperLPR' : (detailRecord.modelType === 'fusion' ?
                    'analyze.modelNameFusion'
                    : (detailRecord.modelType === 'yolov11' ? 'analyze.modelNameYolov11' : 'analyze.modelNameYolo26')))) }}
              </span>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('history.detailTime')">{{ detailRecord.processingTimeMs ?
              detailRecord.processingTimeMs.toFixed(1) : '-' }}ms</el-descriptions-item>
            <el-descriptions-item :label="$t('history.detailDate')">{{ detailRecord.createdAt }}</el-descriptions-item>
            <el-descriptions-item :label="$t('history.detailCount')">{{ detailRecord.detectCount
            }}</el-descriptions-item>
          </el-descriptions>
        </div>
      </div>
    </el-dialog>

    <!-- Feedback Dialog -->
    <el-dialog v-model="showFeedbackDialog" :title="$t('feedback.dialogTitle')" width="500px">
      <p style="margin-bottom: 20px; color: var(--text-secondary); line-height: 1.5;">
        {{ $t('feedback.dialogTips') }}
      </p>
      <el-form label-position="top">
        <el-form-item :label="$t('feedback.labelCorrectPlate')">
          <el-input v-model="feedbackPlate" :placeholder="$t('feedback.placeholderCorrectPlate')" clearable />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showFeedbackDialog = false">{{ $t('history.confirmCancel') }}</el-button>
          <el-button type="primary" @click="submitFeedback" :loading="submittingFeedback">
            {{ submittingFeedback ? $t('feedback.btnWait') : $t('feedback.btnSubmit') }}
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Calendar, Search } from '@element-plus/icons-vue'
import { ElMessageBox } from 'element-plus'
import { useMessage } from '@/composables/useMessage'

const { t } = useI18n()
const message = useMessage()

const loading = ref(true)
const searchQuery = ref('')
const dateRange = ref(null)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const tableData = ref([])
const detailVisible = ref(false)
const detailRecord = ref(null)
const selectedRows = ref([])

const handleSelectionChange = (val) => {
  selectedRows.value = val
}

const batchDeleteRecords = () => {
  ElMessageBox.confirm(
    '确认批量删除选中的识别记录？此操作不可恢复。',
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
      const headers = { 'Authorization': `Bearer ${token}` }
      const promises = selectedRows.value.map(row =>
        fetch(`/api/v1/history/${row.id}`, { method: 'DELETE', headers }).then(res => res.json())
      )
      await Promise.all(promises)
      message.success('批量删除成功')
      fetchData()
    } catch (err) {
      message.error('批量删除时发生错误')
      console.error(err)
    }
  }).catch(() => { })
}

const showFeedbackDialog = ref(false)
const feedbackPlate = ref('')
const submittingFeedback = ref(false)
const currentFeedbackRecord = ref(null)

const fetchData = async () => {
  loading.value = true
  try {
    const params = new URLSearchParams()
    params.append('page', currentPage.value)
    params.append('size', pageSize.value)

    if (searchQuery.value) {
      params.append('keyword', searchQuery.value)
    }
    if (dateRange.value && dateRange.value.length === 2) {
      params.append('startDate', formatDate(dateRange.value[0]))
      params.append('endDate', formatDate(dateRange.value[1]))
    }

    const token = localStorage.getItem('token')
    const headers = {}
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    const res = await fetch(`/api/v1/history/list?${params.toString()}`, { headers })
    const data = await res.json()

    if (data.code === 200) {
      tableData.value = data.data.records
      total.value = data.data.total
    } else {
      message.error(data.message || t('history.fetchFail'))
    }
  } catch (err) {
    message.error(t('history.networkFail'))
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

const openFeedback = (row) => {
  currentFeedbackRecord.value = row
  feedbackPlate.value = row.plateNumber || ''
  showFeedbackDialog.value = true
}

const submitFeedback = async () => {
  if (!feedbackPlate.value.trim()) {
    message.warning('请输入正确的车牌号码')
    return
  }

  submittingFeedback.value = true
  try {
    const token = localStorage.getItem('token')
    const res = await fetch('/api/v1/feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        originalImageUrl: currentFeedbackRecord.value.originalImageUrl,
        resultImageUrl: currentFeedbackRecord.value.resultImageUrl,
        recognizedPlate: currentFeedbackRecord.value.plateNumber,
        modelType: currentFeedbackRecord.value.modelType,
        correctedPlate: feedbackPlate.value.trim()
      })
    })

    const data = await res.json()
    if (data.code === 200) {
      message.success(t('feedback.submitSuccess'))
      showFeedbackDialog.value = false
    } else {
      message.error(data.message || t('feedback.submitFail'))
    }
  } catch (err) {
    message.error(t('feedback.submitFail'))
    console.error(err)
  } finally {
    submittingFeedback.value = false
  }
}

const showDetail = (row) => {
  detailRecord.value = row
  detailVisible.value = true
}

const deleteRecord = async (row) => {
  try {
    await ElMessageBox.confirm(t('history.confirmDelete'), t('history.confirmTitle'), {
      confirmButtonText: t('history.confirmOk'),
      cancelButtonText: t('history.confirmCancel'),
      type: 'warning',
      customClass: 'notion-msgbox'
    })

    const token = localStorage.getItem('token')
    const headers = {}
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    const res = await fetch(`/api/v1/history/${row.id}`, {
      method: 'DELETE',
      headers,
    })
    const data = await res.json()

    if (data.code === 200) {
      message.success(t('history.deleteSuccess'))
      fetchData()
    } else {
      message.error(data.message || t('history.deleteFail'))
    }
  } catch (err) {
    if (err !== 'cancel') {
      message.error(t('history.deleteFail'))
      console.error(err)
    }
  }
}

onMounted(() => {
  fetchData()
})

const tableRowClassName = ({ rowIndex }) => {
  return 'custom-table-row'
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
.history-container {
  max-width: 1200px;
}

.header-section {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 32px;
  flex-wrap: wrap;
  gap: 20px;
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
  margin: 0;
}

.actions-area {
  display: flex;
  gap: 16px;
  align-items: center;
}

.search-input {
  width: 240px;
}

.date-picker {
  width: 320px !important;
}

.action-btn-group {
  display: flex;
  gap: 8px;
  justify-content: center;
  align-items: center;
}

.notion-btn {
  font-weight: 600 !important;
  border: 1px solid rgba(0, 0, 0, 0.1) !important;
  color: #fff !important;
  border-radius: 6px !important;
  transition: all 0.2s ease;
  margin: 0 !important;
}

.primary-btn {
  background-color: #2563eb !important;
}

.primary-btn:hover {
  background-color: #1d4ed8 !important;
}

.warning-btn {
  background-color: #d97706 !important;
}

.warning-btn:hover {
  background-color: #b45309 !important;
}

.danger-btn {
  background-color: #dc2626 !important;
}

.danger-btn:hover {
  background-color: #b91c1c !important;
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

.date-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--text-secondary);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.9rem;
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

.time-cell {
  color: var(--text-secondary);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.thumb-img {
  width: 80px;
  height: 32px;
  border-radius: 4px;
  border: 1px solid var(--border-color);
}

.pagination-wrapper {
  padding: 16px 24px;
  border-top: 1px solid var(--border-color);
  display: flex;
  justify-content: flex-end;
  background: white;
}

/* Detail dialog styles */
.detail-content {
  padding: 0 8px;
}

.detail-images {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.detail-image-box {
  text-align: center;
}

.detail-image-box h4 {
  margin-bottom: 12px;
  font-weight: 600;
  color: var(--text-primary);
}

.detail-info {
  margin-top: 16px;
}
</style>
