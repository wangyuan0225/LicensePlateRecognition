<template>
  <div class="history-container page-container">
    <div class="header-section">
      <div class="title-area">
        <h1 class="page-title">{{ $t('history.title') }}</h1>
        <p class="page-subtitle">{{ $t('history.subtitle') }}</p>
      </div>
      <div class="actions-area">
        <el-input v-model="searchQuery" :placeholder="$t('history.searchPlaceholder')" prefix-icon="Search"
          class="search-input" clearable @clear="fetchData" @keyup.enter="fetchData" />
        <el-date-picker v-model="dateRange" type="daterange" range-separator="-"
          :start-placeholder="$t('history.startDate')" :end-placeholder="$t('history.endDate')" class="date-picker"
          @change="fetchData" />
        <el-button type="primary" icon="Search" @click="fetchData">搜索</el-button>
      </div>
    </div>

    <div class="notion-card table-wrapper">
      <el-table :data="tableData" style="width: 100%" v-loading="loading" :row-class-name="tableRowClassName">
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
            <el-tag size="large" type="success" effect="plain" class="plate-tag">
              {{ scope.row.plateNumber }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column prop="plateType" label="车牌属性" min-width="120" align="center">
          <template #default="scope">
            <span class="model-cell">{{ scope.row.plateType || '-' }}</span>
          </template>
        </el-table-column>

        <el-table-column prop="modelType" :label="$t('history.colModel')" min-width="140" align="center">
          <template #default="scope">
            <el-tag :type="scope.row.modelType === 'yolov8' ? 'warning' : 'success'" size="small" effect="light">
              {{ scope.row.modelType === 'yolov8' ? 'YOLOv8' : 'YOLO26' }}
            </el-tag>
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

        <el-table-column fixed="right" :label="$t('history.colActions')" width="140" align="center">
          <template #default="scope">
            <el-button link type="primary" size="small" @click="showDetail(scope.row)">{{ $t('history.actionDetails')
            }}</el-button>
            <el-button link type="danger" size="small" @click="deleteRecord(scope.row)">{{ $t('history.actionDelete')
            }}</el-button>
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
    <el-dialog v-model="detailVisible" title="识别详情" width="800px">
      <div v-if="detailRecord" class="detail-content">
        <div class="detail-images">
          <div class="detail-image-box">
            <h4>原始图像</h4>
            <el-image :src="detailRecord.originalImageUrl" fit="contain" style="width: 100%; height: 300px;" />
          </div>
          <div class="detail-image-box">
            <h4>检测结果图像</h4>
            <el-image :src="detailRecord.resultImageUrl" fit="contain" style="width: 100%; height: 300px;" />
          </div>
        </div>
        <div class="detail-info">
          <el-descriptions :column="2" border>
            <el-descriptions-item label="车牌号码">{{ detailRecord.plateNumber }}</el-descriptions-item>
            <el-descriptions-item label="车牌属性">{{ detailRecord.plateType || '-' }}</el-descriptions-item>
            <el-descriptions-item label="算法模型">
              <el-tag :type="detailRecord.modelType === 'yolov8' ? 'warning' : 'success'" size="small">
                {{ detailRecord.modelType === 'yolov8' ? 'YOLOv8' : 'YOLO26' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="识别耗时">{{ detailRecord.processingTimeMs ?
              detailRecord.processingTimeMs.toFixed(1) : '-' }}ms</el-descriptions-item>
            <el-descriptions-item label="识别时间">{{ detailRecord.createdAt }}</el-descriptions-item>
            <el-descriptions-item label="检测数量">{{ detailRecord.detectCount }}</el-descriptions-item>
          </el-descriptions>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Calendar, Search } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'

const loading = ref(true)
const searchQuery = ref('')
const dateRange = ref(null)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const tableData = ref([])
const detailVisible = ref(false)
const detailRecord = ref(null)

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
      ElMessage.error(data.message || '获取历史记录失败')
    }
  } catch (err) {
    ElMessage.error('网络请求失败，请检查后端是否启动')
    console.error(err)
  } finally {
    loading.value = false
  }
}

const formatDate = (date) => {
  if (!date) return ''
  const d = new Date(date)
  return d.toISOString().split('T')[0]
}

const showDetail = (row) => {
  detailRecord.value = row
  detailVisible.value = true
}

const deleteRecord = async (row) => {
  try {
    await ElMessageBox.confirm('确认删除该条记录？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
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
      ElMessage.success('删除成功')
      fetchData()
    } else {
      ElMessage.error(data.message || '删除失败')
    }
  } catch (err) {
    if (err !== 'cancel') {
      ElMessage.error('删除失败')
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

.plate-tag {
  font-weight: 700;
  letter-spacing: 1px;
}

.model-cell {
  background: var(--bg-secondary);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.85rem;
  border: 1px solid var(--border-color);
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
