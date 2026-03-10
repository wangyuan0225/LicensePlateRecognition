<template>
    <div class="admin-feedback-container">
        <div class="page-header" style="display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <h1 class="page-title">{{ $t('app.adminFeedback') }}</h1>
                <p class="page-subtitle">View and filter all users' submitted feedbacks.</p>
            </div>
            <div class="actions-area" v-if="selectedRows.length > 0">
                <el-button class="notion-btn approve-btn" @click="batchHandleStatus('APPROVED')">
                    批量审批
                </el-button>
                <el-button class="notion-btn reject-btn" @click="batchHandleStatus('REJECTED')">
                    批量驳回
                </el-button>
            </div>
        </div>

        <el-card class="filter-card" shadow="never">
            <el-form :inline="true" :model="filters" class="filter-form" @submit.prevent="handleSearch">
                <el-form-item :label="$t('app.filterUser')" style="margin-bottom: 0;">
                    <el-select v-model="filters.userId" :placeholder="$t('app.filterAllUsers')" clearable
                        style="width: 200px;">
                        <el-option v-for="user in users" :key="user.id" :label="user.username" :value="user.id" />
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
                    <el-button type="primary" @click="handleSearch" :icon="Search">{{ $t('history.searchBtn')
                        }}</el-button>
                </el-form-item>
            </el-form>
        </el-card>

        <div class="notion-card table-wrapper" v-loading="loading">
            <el-table :data="tableData" style="width: 100%" max-height="600" @selection-change="handleSelectionChange"
                :empty-text="$t('feedback.noData')">
                <el-table-column type="selection" width="55" align="center" />
                <el-table-column prop="username" :label="$t('feedback.colUser')" min-width="120" />

                <el-table-column :label="$t('feedback.colOriginal')" width="100" align="center">
                    <template #default="scope">
                        <el-image style="width: 60px; height: 40px; border-radius: 4px;"
                            :src="scope.row.originalImageUrl" :preview-src-list="[scope.row.originalImageUrl]"
                            fit="cover" lazy :preview-teleported="true">
                            <template #error>
                                <div class="image-slot"><el-icon>
                                        <Picture />
                                    </el-icon></div>
                            </template>
                        </el-image>
                    </template>
                </el-table-column>

                <el-table-column :label="$t('feedback.colResult')" width="100" align="center">
                    <template #default="scope">
                        <el-image style="width: 60px; height: 40px; border-radius: 4px;" :src="scope.row.resultImageUrl"
                            :preview-src-list="[scope.row.resultImageUrl]" fit="cover" lazy :preview-teleported="true">
                            <template #error>
                                <div class="image-slot"><el-icon>
                                        <Picture />
                                    </el-icon></div>
                            </template>
                        </el-image>
                    </template>
                </el-table-column>

                <el-table-column prop="recognizedPlate" :label="$t('feedback.colRecognized')" min-width="140"
                    align="center">
                    <template #default="scope">
                        <span class="custom-tag" :style="getPlateColorStyle(scope.row.recognizedPlate)">
                            {{ scope.row.recognizedPlate || '未识别' }}
                        </span>
                    </template>
                </el-table-column>

                <el-table-column prop="correctedPlate" :label="$t('feedback.colCorrected')" min-width="140"
                    align="center">
                    <template #default="scope">
                        <span class="custom-tag" :style="getPlateColorStyle(scope.row.correctedPlate)">
                            {{ scope.row.correctedPlate }}
                        </span>
                    </template>
                </el-table-column>

                <el-table-column :label="$t('feedback.colModel')" min-width="120" align="center">
                    <template #default="scope">
                        <span class="custom-tag">
                            {{ formatModelName(scope.row.modelType) }}
                        </span>
                    </template>
                </el-table-column>

                <el-table-column :label="$t('feedback.colStatus')" min-width="120" align="center">
                    <template #default="scope">
                        <span class="custom-tag" :style="getStatusColorStyle(scope.row.status)">
                            {{ getStatusText(scope.row.status) }}
                        </span>
                    </template>
                </el-table-column>

                <el-table-column prop="createdAt" :label="$t('feedback.colTime')" width="180" align="center">
                    <template #default="scope">
                        <span class="date-cell">{{ scope.row.createdAt }}</span>
                    </template>
                </el-table-column>

                <el-table-column :label="$t('history.colActions')" width="160" align="center" fixed="right">
                    <template #default="scope">
                        <div class="action-btn-group">
                            <el-button class="notion-btn approve-btn" size="small"
                                @click="handleStatusChange(scope.row.id, 'APPROVED')">
                                {{ $t('app.actionApprove') }}
                            </el-button>
                            <el-button class="notion-btn reject-btn" size="small"
                                @click="handleStatusChange(scope.row.id, 'REJECTED')">
                                {{ $t('app.actionReject') }}
                            </el-button>
                        </div>
                    </template>
                </el-table-column>
            </el-table>
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
const users = ref([])
const selectedRows = ref([])

const handleSelectionChange = (val) => {
    selectedRows.value = val
}

const batchHandleStatus = async (newStatus) => {
    if (selectedRows.value.length === 0) return
    try {
        const token = localStorage.getItem('token')
        const promises = selectedRows.value.map(row =>
            axios.put(`/api/v1/admin/feedback/${row.id}/status`,
                { status: newStatus },
                { headers: { 'Authorization': `Bearer ${token}` } }
            )
        )
        await Promise.all(promises)
        ElMessage.success('批量操作成功')
        fetchFeedback()
    } catch (e) {
        ElMessage.error('批量操作失败')
    }
}

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

const fetchFeedback = async () => {
    loading.value = true
    try {
        const token = localStorage.getItem('token')
        const params = {
            userId: filters.value.userId || undefined,
            modelType: filters.value.modelType || undefined
        }

        const res = await axios.get('/api/v1/admin/feedback', {
            params,
            headers: { 'Authorization': `Bearer ${token}` }
        })

        if (res.data.code === 200) {
            tableData.value = res.data.data
        } else {
            ElMessage.error(res.data.msg || t('feedback.fetchFail'))
        }
    } catch (error) {
        ElMessage.error(error.response?.data?.msg || t('history.networkFail'))
    } finally {
        loading.value = false
    }
}

const handleSearch = () => {
    fetchFeedback()
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

const handleStatusChange = async (id, newStatus) => {
    try {
        const token = localStorage.getItem('token')
        const res = await axios.put(`/api/v1/admin/feedback/${id}/status`,
            { status: newStatus },
            { headers: { 'Authorization': `Bearer ${token}` } }
        )
        if (res.data.code === 200) {
            ElMessage.success('状态已更新')
            fetchFeedback() // Reload table
        } else {
            ElMessage.error(res.data.msg || '更新失败')
        }
    } catch (e) {
        ElMessage.error('网络请求失败')
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

onMounted(() => {
    fetchUsers()
    fetchFeedback()
})
</script>

<style scoped>
.admin-feedback-container {
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

.date-cell {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.9rem;
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

.approve-btn {
    background-color: #16a34a !important;
}

.approve-btn:hover {
    background-color: #15803d !important;
}

.reject-btn {
    background-color: #dc2626 !important;
}

.reject-btn:hover {
    background-color: #b91c1c !important;
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
</style>
