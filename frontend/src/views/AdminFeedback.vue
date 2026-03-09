<template>
    <div class="admin-feedback-container">
        <div class="page-header">
            <div>
                <h1 class="page-title">{{ $t('app.adminFeedback') }}</h1>
                <p class="page-subtitle">View and filter all users' submitted feedbacks.</p>
            </div>
        </div>

        <el-card class="filter-card" shadow="never">
            <el-form :inline="true" :model="filters" class="filter-form" @submit.prevent="handleSearch">
                <el-form-item label="User" style="margin-bottom: 0;">
                    <el-select v-model="filters.userId" placeholder="All Users" clearable style="width: 200px;">
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
                    </el-select>
                </el-form-item>

                <el-form-item style="margin-bottom: 0;">
                    <el-button type="primary" @click="handleSearch" :icon="Search">{{ $t('history.searchBtn') }}</el-button>
                </el-form-item>
            </el-form>
        </el-card>

        <el-card class="table-card" shadow="sm" v-loading="loading">
            <el-table :data="tableData" style="width: 100%" max-height="600" :empty-text="$t('feedback.noData')">
                <el-table-column prop="username" :label="$t('feedback.colUser')" min-width="120" />
                
                <el-table-column :label="$t('feedback.colOriginal')" width="100" align="center">
                    <template #default="scope">
                        <el-image
                            style="width: 60px; height: 40px; border-radius: 4px;"
                            :src="scope.row.originalImageUrl"
                            :preview-src-list="[scope.row.originalImageUrl]"
                            fit="cover"
                            lazy
                            :preview-teleported="true"
                        >
                            <template #error>
                                <div class="image-slot"><el-icon><Picture /></el-icon></div>
                            </template>
                        </el-image>
                    </template>
                </el-table-column>

                <el-table-column :label="$t('feedback.colResult')" width="100" align="center">
                    <template #default="scope">
                        <el-image
                            style="width: 60px; height: 40px; border-radius: 4px;"
                            :src="scope.row.resultImageUrl"
                            :preview-src-list="[scope.row.resultImageUrl]"
                            fit="cover"
                            lazy
                            :preview-teleported="true"
                        >
                            <template #error>
                                <div class="image-slot"><el-icon><Picture /></el-icon></div>
                            </template>
                        </el-image>
                    </template>
                </el-table-column>

                <el-table-column prop="recognizedPlate" :label="$t('feedback.colRecognized')" min-width="140">
                    <template #default="scope">
                        <el-tag size="large" type="info" effect="plain" class="plate-tag">
                            {{ scope.row.recognizedPlate || '未识别' }}
                        </el-tag>
                    </template>
                </el-table-column>
                
                <el-table-column prop="correctedPlate" :label="$t('feedback.colCorrected')" min-width="140">
                    <template #default="scope">
                        <el-tag size="large" type="success" effect="plain" class="plate-tag corrected-tag">
                            {{ scope.row.correctedPlate }}
                        </el-tag>
                    </template>
                </el-table-column>

                <el-table-column :label="$t('feedback.colModel')" min-width="120">
                    <template #default="scope">
                        <el-tag size="small" type="info">
                            {{ formatModelName(scope.row.modelType) }}
                        </el-tag>
                    </template>
                </el-table-column>

                <el-table-column prop="createdAt" :label="$t('feedback.colTime')" width="180" />
            </el-table>
        </el-card>
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
        'fusion': t('analyze.modelNameFusion')
    }
    return map[modelKey] || modelKey
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

.table-card {
    border-radius: 12px;
    border: 1px solid var(--border-color);
}

.plate-tag {
    font-weight: bold;
    font-family: monospace;
    font-size: 14px;
}

.corrected-tag {
    color: #16a34a !important;
    border-color: #16a34a !important;
    background-color: #f0fdf4 !important;
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
