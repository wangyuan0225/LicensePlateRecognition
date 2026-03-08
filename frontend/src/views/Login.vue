<template>
  <div class="auth-container page-container">
    <div class="auth-card notion-card">
      <div class="auth-header">
        <h2 class="auth-title">{{ isLogin ? $t('login.welcomeLogin') : $t('login.welcomeRegister') }}</h2>
        <p class="auth-subtitle">
          {{ isLogin ? $t('login.loginSubtitle') : $t('login.registerSubtitle') }}
        </p>
      </div>

      <el-form ref="formRef" :model="form" :rules="rules" label-position="top" class="auth-form"
        @keyup.enter="handleSubmit">
        <transition name="el-fade-in-linear">
          <el-form-item v-if="!isLogin" :label="$t('login.username')" prop="username">
            <el-input v-model="form.username" :placeholder="$t('login.usernamePlaceholder')" prefix-icon="User" />
          </el-form-item>
        </transition>

        <el-form-item :label="$t('login.email')" prop="email">
          <el-input v-model="form.email" :placeholder="$t('login.emailPlaceholder')" prefix-icon="Message" />
        </el-form-item>

        <el-form-item :label="$t('login.password')" prop="password">
          <el-input v-model="form.password" type="password" :placeholder="$t('login.passwordPlaceholder')"
            prefix-icon="Lock" show-password />
        </el-form-item>

        <el-form-item v-if="isLogin">
          <div class="form-options">
            <el-checkbox v-model="form.rememberMe">{{ $t('login.rememberMe') }}</el-checkbox>
            <el-link type="primary" :underline="false">{{ $t('login.forgotPassword') }}</el-link>
          </div>
        </el-form-item>

        <el-button type="primary" class="submit-btn" :loading="loading" @click="handleSubmit">
          {{ isLogin ? $t('login.loginBtn') : $t('login.registerBtn') }}
        </el-button>
      </el-form>

      <div class="auth-footer">
        <span class="text-secondary">
          {{ isLogin ? $t('login.noAccount') : $t('login.hasAccount') }}
        </span>
        <el-link type="primary" @click="toggleMode" :underline="false">
          {{ isLogin ? $t('login.registerBtn') : $t('login.loginBtn') }}
        </el-link>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'

const router = useRouter()
const { t } = useI18n()
const formRef = ref(null)

const isLogin = ref(true)
const loading = ref(false)

const form = reactive({
  username: '',
  email: '',
  password: '',
  rememberMe: false
})

const rules = computed(() => ({
  username: [
    { required: true, message: t('login.ruleUsernameRequired'), trigger: 'blur' },
    { min: 3, max: 20, message: t('login.ruleUsernameLength'), trigger: 'blur' },
  ],
  email: [
    { required: true, message: t('login.ruleEmailRequired'), trigger: 'blur' },
    { type: 'email', message: t('login.ruleEmailFormat'), trigger: ['blur', 'change'] },
  ],
  password: [
    { required: true, message: t('login.rulePasswordRequired'), trigger: 'blur' },
    { min: 6, max: 20, message: t('login.rulePasswordLength'), trigger: 'blur' },
  ]
}))

const toggleMode = () => {
  isLogin.value = !isLogin.value
  formRef.value?.resetFields()
}

const handleSubmit = async () => {
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (valid) {
      loading.value = true

      try {
        if (isLogin.value) {
          // Login
          const res = await fetch('/api/v1/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              email: form.email,
              password: form.password,
            }),
          })
          const data = await res.json()

          if (data.code === 200) {
            localStorage.setItem('token', data.data.token)
            localStorage.setItem('user', JSON.stringify(data.data.user))
            ElMessage.success(t('login.loginSuccess'))
            router.push('/analyze')
          } else {
            ElMessage.error(data.message || t('login.loginFail'))
          }
        } else {
          // Register
          const res = await fetch('/api/v1/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              username: form.username,
              email: form.email,
              password: form.password,
            }),
          })
          const data = await res.json()

          if (data.code === 201) {
            ElMessage.success(t('login.registerSuccess'))
            isLogin.value = true
            formRef.value?.resetFields()
          } else {
            ElMessage.error(data.message || t('login.registerFail'))
          }
        }
      } catch (err) {
        ElMessage.error(t('login.networkFail'))
        console.error(err)
      } finally {
        loading.value = false
      }
    }
  })
}
</script>

<style scoped>
.auth-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 160px);
}

.auth-card {
  width: 100%;
  max-width: 440px;
  padding: 40px;
}

.auth-header {
  text-align: center;
  margin-bottom: 32px;
}

.auth-title {
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--text-primary);
  letter-spacing: -0.02em;
}

.auth-subtitle {
  color: var(--text-secondary);
  font-size: 0.95rem;
}

.auth-form {
  margin-bottom: 24px;
}

:deep(.el-form-item__label) {
  font-weight: 500;
  color: var(--text-primary);
  padding-bottom: 4px;
}

:deep(.el-input__wrapper) {
  box-shadow: 0 0 0 1px var(--border-color) inset;
  padding: 4px 12px;
}

:deep(.el-input__wrapper.is-focus) {
  box-shadow: 0 0 0 1px var(--primary-color) inset !important;
}

.form-options {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.submit-btn {
  width: 100%;
  height: 44px;
  font-size: 1rem;
  font-weight: 500;
  margin-top: 8px;
}

.auth-footer {
  text-align: center;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.text-secondary {
  color: var(--text-secondary);
}
</style>
