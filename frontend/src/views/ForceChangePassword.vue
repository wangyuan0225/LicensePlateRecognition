<template>
    <div class="force-pwd-container">
        <div class="force-pwd-box">
            <h2 class="title">{{ $t('forcePwd.title') }}</h2>
            <p class="subtitle">{{ $t('forcePwd.subtitle') }}</p>

            <el-form :model="form" :rules="rules" ref="formRef" size="large">
                <el-form-item prop="oldPassword">
                    <el-input v-model="form.oldPassword" :placeholder="$t('forcePwd.oldPasswordPlaceholder')" show-password clearable>
                        <template #prefix>
                            <el-icon><Lock /></el-icon>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item prop="newPassword">
                    <el-input v-model="form.newPassword" :placeholder="$t('forcePwd.newPasswordPlaceholder')" show-password clearable>
                        <template #prefix>
                            <el-icon><Key /></el-icon>
                        </template>
                    </el-input>
                </el-form-item>

                <el-button type="primary" class="w-full mt-4" @click="onSubmit" :loading="loading">
                    {{ $t('forcePwd.submitBtn') }}
                </el-button>
            </el-form>
        </div>
    </div>
</template>

<script setup>
import { reactive, ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Lock, Key } from '@element-plus/icons-vue'
import axios from 'axios'

const router = useRouter()
const { t } = useI18n()

const formRef = ref(null)
const loading = ref(false)

const form = reactive({
    oldPassword: '',
    newPassword: ''
})

const rules = computed(() => ({
    oldPassword: [
        { required: true, message: t('forcePwd.ruleOldPasswordRequired'), trigger: 'blur' }
    ],
    newPassword: [
        { required: true, message: t('forcePwd.ruleNewPasswordRequired'), trigger: 'blur' },
        { min: 6, max: 20, message: t('forcePwd.rulePasswordLength'), trigger: 'blur' }
    ]
}))

const onSubmit = () => {
    formRef.value.validate(async (valid) => {
        if (!valid) return
        loading.value = true
        try {
            const token = localStorage.getItem('token')
            const res = await axios.post('/api/v1/auth/force-change-password', form, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
            if (res.data.code === 200) {
                ElMessage.success(t('forcePwd.successMsg'))
                // Log out locally
                localStorage.removeItem('token')
                localStorage.removeItem('user')
                router.push('/login')
            } else {
                ElMessage.error(res.data.msg || 'Submit failed')
            }
        } catch (error) {
            ElMessage.error(error.response?.data?.msg || 'Error')
        } finally {
            loading.value = false
        }
    })
}
</script>

<style scoped>
.force-pwd-container {
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.force-pwd-box {
    width: 400px;
    padding: 40px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
}

.title {
    margin: 0 0 10px 0;
    font-size: 24px;
    text-align: center;
    color: #303133;
}

.subtitle {
    margin: 0 0 30px 0;
    font-size: 14px;
    text-align: center;
    color: #909399;
}

.w-full {
    width: 100%;
}

.mt-4 {
    margin-top: 1rem;
}
</style>
