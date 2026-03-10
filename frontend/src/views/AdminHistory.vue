<template>
    <div class="admin-history-container">
        <div class="page-header">
            <div>
                <h1 class="page-title">{{ $t('app.adminHistory') }}</h1>
                <p class="page-subtitle">{{ $t('history.subtitle') }}</p>
            </div>
        </div>

        <el-card class="filter-card" shadow="never">
            <el-form :inline="true" :model="filters" class="filter-form" @submit.prevent="handleSearch">
                <el-form-item :label="$t('app.filterUser')" style="margin-bottom: 0;">
                    <el-select v-model="filters.userId" :placeholder="$t('app.filterAllUsers')" clearable style="width: 200px;">
                        <el-option
                            v-for="user in users"
                            :key="user.id"
                            :label="user.username"
                            :value="user.id"
                        />
                    </el-select>
                </el-form-item>
                
                <el-form-item :label="$t('analyze.modelLabel')" style="margin-bottom: 0;">
                    <el-select v-model="filters.modelType" placeholder="All Models" clearable style="width: 200px;">
                        <el-option :label="$t('analyze.modelNameYolo26')" value="yolo26" />
                        <el-option :label="$t('analyze.modelNameYolov8')" value="yolov8" />
                        <el-option :label="$t('analyze.modelNameHyperLPR')" value="hyperlpr" />
                        <el-option :label="$t('analyze.modelNameFusion')" value="fusion" />
                        <el-option :label="$t('analyze.modelNameYolov11')" value="yolov11" />
                    </el-select>
                </el-form-item>

                <el-form-item style="margin-bottom: 0;">
                    <el-button type="primary" @click="handleSearch" :icon="Search">{{ $t('history.searchBtn') }}</el-button>
                </el-form-item>
            </el-form>
        </el-card>

        <div class="notion-card table-wrapper" v-loading="loading">
            <el-table :data="tableData" style="width: 100%" max-height="600" empty-text="No data">
                <!-- Username -->
                <el-table-column prop="username" :label="$t('history.colUser')" min-width="120" />
                
                <!-- Thumbnail -->
                <el-table-column :label="$t('history.colThumb')" width="100" align="center">
                    <template #default="scope">
                        <el-image
                            style="width: 60px; height: 40px; border-radius: 4px;"
                            :src="scope.row.thumbnailUrl"
                            :preview-src-list="[scope.row.originalImageUrl]"
                            fit="cover"
                            lazy
                            :preview-teleported="true"
                        >
                            <template #error>
                                <div class="image-slot">
                                    <el-icon><Picture /></el-icon>
                                </div>
                            </template>
                        </el-image>
                    </template>
                </el-table-column>

                <el-table-column prop="plateNumber" :label="$t('history.colPlate')" min-width="140" align="center">
                    <template #default="scope">
                        <span class="custom-tag" :style="getPlateColorStyle(scope.row.plateType)">
                            {{ scope.row.plateNumber || '未识别' }}
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

                <el-table-column :label="$t('history.colModel')" min-width="120" align="center">
                    <template #default="scope">
                        <span class="custom-tag">
                            {{ formatModelName(scope.row.modelType) }}
                        </span>
                    </template>
                </el-table-column>

                <el-table-column prop="processingTimeMs" :label="$t('history.colTime')" min-width="120" align="center">
                    <template #default="scope">
                        <span class="time-cell">{{ scope.row.processingTimeMs ? scope.row.processingTimeMs.toFixed(1) : '-' }}ms</span>
                    </template>
                </el-table-column>

                <el-table-column prop="createdAt" :label="$t('history.colDate')" width="200" align="center">
                    <template #default="scope">
                        <span class="date-cell">{{ scope.row.createdAt }}</span>
                    </template>
                </el-table-column>
            </el-table>

            <div class="pagination-container">
                <el-pagination
                    v-model:current-page="currentPage"
                    v-model:page-size="pageSize"
                    :page-sizes="[10, 20, 50, 100]"
                    background
                    layout="total, sizes, prev, pager, next, jumper"
                    :total="totalCount"
                    @size-change="handleSizeChange"
                    @current-change="handleCurrentChange"
                />
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Search, Picture } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const loading = ref(false)
const tableData = ref([])
const totalCount = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)

const users = ref([])

const filters = ref({
    userId: null,
    modelType: ''
})

const fetchUsers = async () => {
    try {
        const token = localStorage.getItem('token')
        const res = await axios.get('/api/v1/admin/users', {
            headers: { 'Authorization': `Bearer ${token}` }
        })
        if (res.data.code === 200) {
            users.value = res.data.data
        }
    } catch (error) {
        console.error('Fetch users error', error)
    }
}

const fetchHistory = async () => {
    loading.value = true
    try {
        const token = localStorage.getItem('token')
        const params = {
            page: currentPage.value,
            size: pageSize.value,
            userId: filters.value.userId || undefined,
            modelType: filters.value.modelType || undefined
        }
        
        const res = await axios.get('/api/v1/admin/history', {
            params,
            headers: { 'Authorization': `Bearer ${token}` }
        })

        if (res.data.code === 200) {
            tableData.value = res.data.data.records
            totalCount.value = res.data.data.total
        } else {
            ElMessage.error(res.data.msg || t('history.fetchFail'))
        }
    } catch (error) {
        ElMessage.error(error.response?.data?.msg || t('history.networkFail'))
    } finally {
        loading.value = false
    }
}

const handleSearch = () => {
    currentPage.value = 1
    fetchHistory()
}

const handleSizeChange = (val) => {
    pageSize.value = val
    fetchHistory()
}

const handleCurrentChange = (val) => {
    currentPage.value = val
    fetchHistory()
}

const formatModelName = (modelKey) => {
    const map = {
        'yolo26': t('analyze.modelNameYolo26'),
        'yolov8': t('analyze.modelNameYolov8'),
        'hyperlpr': t('analyze.modelNameHyperLPR'),
        'fusion': t('analyze.modelNameFusion'),
        'yolov11': t('analyze.modelNameYolov11')
    }
    return map[modelKey] || modelKey
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

onMounted(() => {
    fetchUsers()
    fetchHistory()
})
</script>

<style scoped>
.admin-history-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 30px;
}

.page-header {
    margin-bottom: 24px;
}

.page-title {
    font-size: 28px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 8px 0;
}

.page-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0;
}

.filter-card {
    margin-bottom: 24px;
    border-radius: 8px;
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
}

.filter-form {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-end;
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

.time-cell {
    color: var(--text-secondary);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.date-cell {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.9rem;
}

.image-slot {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 100%;
    background: #f5f7fa;
    color: #909399;
}

.pagination-container {
    display: flex;
    justify-content: flex-end;
    margin-top: 24px;
    margin-bottom: 24px;
    margin-right: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--border-light);
}
</style>
