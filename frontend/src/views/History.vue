<template>
  <div class="history-container page-container">
    <div class="header-section">
      <div class="title-area">
        <h1 class="page-title">{{ $t('history.title') }}</h1>
        <p class="page-subtitle">{{ $t('history.subtitle') }}</p>
      </div>
      <div class="actions-area">
        <el-input v-model="searchQuery" :placeholder="$t('history.searchPlaceholder')" prefix-icon="Search"
          class="search-input" clearable />
        <el-date-picker v-model="dateRange" type="daterange" range-separator="-"
          :start-placeholder="$t('history.startDate')" :end-placeholder="$t('history.endDate')" class="date-picker" />
        <el-button type="primary" icon="Download">{{ $t('history.exportBtn') }}</el-button>
      </div>
    </div>

    <div class="notion-card table-wrapper">
      <el-table :data="filteredData" style="width: 100%" v-loading="loading" :row-class-name="tableRowClassName">
        <el-table-column prop="date" :label="$t('history.colDate')" min-width="220" align="center">
          <template #default="scope">
            <span class="date-cell">
              <el-icon>
                <Calendar />
              </el-icon>
              {{ scope.row.date }}
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

        <el-table-column prop="model" :label="$t('history.colModel')" min-width="180" align="center">
          <template #default="scope">
            <span class="model-cell">{{ scope.row.model }}</span>
          </template>
        </el-table-column>

        <el-table-column prop="confidence" :label="$t('history.colConfidence')" min-width="140">
          <template #default="scope">
            <span class="confidence-cell" :class="{ 'high-conf': scope.row.confidence >= 0.95 }">
              {{ (scope.row.confidence * 100).toFixed(1) }}%
            </span>
          </template>
        </el-table-column>

        <el-table-column prop="timeTaken" :label="$t('history.colTime')" min-width="120">
          <template #default="scope">
            <span class="time-cell">{{ scope.row.timeTaken }}ms</span>
          </template>
        </el-table-column>

        <el-table-column :label="$t('history.colThumb')" width="140" align="center">
          <template #default="scope">
            <el-image :src="scope.row.thumbnail" class="thumb-img" :preview-src-list="[scope.row.thumbnail]"
              preview-teleported fit="cover" />
          </template>
        </el-table-column>

        <el-table-column fixed="right" :label="$t('history.colActions')" width="140" align="center">
          <template #default="scope">
            <el-button link type="primary" size="small">{{ $t('history.actionDetails') }}</el-button>
            <el-button link type="danger" size="small">{{ $t('history.actionDelete') }}</el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-wrapper">
        <el-pagination v-model:current-page="currentPage" v-model:page-size="pageSize" :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper" :total="mockData.length" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Calendar, Search, Download } from '@element-plus/icons-vue'

const loading = ref(true)
const searchQuery = ref('')
const dateRange = ref(null)
const currentPage = ref(1)
const pageSize = ref(10)

// Mock Data
const mockData = ref([
  { id: 1, date: '2026-02-24 14:30:22', plateNumber: '沪A·88888', model: 'YOLOv8 Fast', confidence: 0.985, timeTaken: 124, thumbnail: 'https://placehold.co/100x40/f2f2f0/333?text=Plate' },
  { id: 2, date: '2026-02-24 13:15:05', plateNumber: '京B·12345', model: 'YOLOv8 High Acc', confidence: 0.992, timeTaken: 256, thumbnail: 'https://placehold.co/100x40/f2f2f0/333?text=Plate' },
  { id: 3, date: '2026-02-23 09:45:11', plateNumber: '粤C·AB345', model: 'ResNet50 Hybrid', confidence: 0.945, timeTaken: 310, thumbnail: 'https://placehold.co/100x40/f2f2f0/333?text=Plate' },
  { id: 4, date: '2026-02-23 08:20:00', plateNumber: '苏D·99X99', model: 'YOLOv8 Fast', confidence: 0.971, timeTaken: 118, thumbnail: 'https://placehold.co/100x40/f2f2f0/333?text=Plate' },
  { id: 5, date: '2026-02-22 16:55:33', plateNumber: '浙E·1827Z', model: 'YOLOv8 Fast', confidence: 0.988, timeTaken: 120, thumbnail: 'https://placehold.co/100x40/f2f2f0/333?text=Plate' },
])

onMounted(() => {
  setTimeout(() => {
    loading.value = false
  }, 600)
})

const filteredData = computed(() => {
  let filtered = mockData.value

  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(item =>
      item.plateNumber.toLowerCase().includes(query)
    )
  }

  // Implement pagination slice here in a real app, currently showing all filtered
  return filtered
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
  /* Remove padding for edge-to-edge table */
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

.confidence-cell {
  font-weight: 500;
}

.confidence-cell.high-conf {
  color: var(--success-color);
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
</style>
