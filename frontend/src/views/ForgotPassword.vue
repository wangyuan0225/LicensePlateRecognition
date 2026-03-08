<template>
  <div class="auth-container page-container">
    <div class="auth-card notion-card">
      <div class="auth-header">
        <h2 class="auth-title">{{ $t('forgotPwd.title') }}</h2>
        <p class="auth-subtitle">{{ $t('forgotPwd.subtitle') }}</p>
      </div>

      <el-form ref="formRef" :model="form" :rules="rules" label-position="top" class="auth-form">
        <!-- 邮箱 -->
        <el-form-item :label="$t('forgotPwd.email')" prop="email">
          <el-input v-model="form.email" :placeholder="$t('forgotPwd.emailPlaceholder')" prefix-icon="Message" />
        </el-form-item>

        <!-- 验证码 -->
        <el-form-item :label="$t('forgotPwd.verifyCode')" prop="code">
          <div class="code-row">
            <el-input v-model="form.code" :placeholder="$t('forgotPwd.verifyCodePlaceholder')"
              prefix-icon="Key" maxlength="6" />
            <el-button
              type="primary" plain
              :disabled="codeCooldown > 0 || !form.email"
              :loading="sendingCode"
              @click="sendCode"
              class="send-code-btn">
              {{ codeCooldown > 0 ? `${codeCooldown}s` : (codeSent ? $t('forgotPwd.sendCodeAgain') : $t('forgotPwd.sendCode')) }}
            </el-button>
          </div>
        </el-form-item>

        <!-- 新密码 -->
        <el-form-item :label="$t('forgotPwd.newPassword')" prop="newPassword">
          <el-input v-model="form.newPassword" type="password"
            :placeholder="$t('forgotPwd.newPasswordPlaceholder')" prefix-icon="Lock" show-password />
        </el-form-item>

        <el-button type="primary" class="submit-btn" :loading="loading" @click="handleSubmit">
          {{ $t('forgotPwd.submitBtn') }}
        </el-button>
      </el-form>

      <div class="auth-footer">
        <el-link type="primary" @click="$router.push('/login')" :underline="false">
          ← {{ $t('forgotPwd.backToLogin') }}
        </el-link>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useMessage } from '@/composables/useMessage'

const router = useRouter()
const { t } = useI18n()
const message = useMessage()
const formRef = ref(null)
const loading = ref(false)
const sendingCode = ref(false)
const codeSent = ref(false)
const codeCooldown = ref(0)
let cooldownTimer = null

const form = reactive({ email: '', code: '', newPassword: '' })

const rules = computed(() => ({
  email: [
    { required: true, message: t('forgotPwd.ruleEmailRequired'), trigger: 'blur' },
    { type: 'email', message: t('forgotPwd.ruleEmailFormat'), trigger: ['blur', 'change'] },
  ],
  code: [
    { required: true, message: t('forgotPwd.ruleCodeRequired'), trigger: 'blur' },
    { len: 6, message: t('forgotPwd.ruleCodeLength'), trigger: 'blur' },
  ],
  newPassword: [
    { required: true, message: t('forgotPwd.rulePasswordRequired'), trigger: 'blur' },
    { min: 6, max: 20, message: t('forgotPwd.rulePasswordLength'), trigger: 'blur' },
  ],
}))

const sendCode = async () => {
  if (!form.email) return
  sendingCode.value = true
  try {
    const res = await fetch('/api/v1/auth/send-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: form.email, type: 'reset' }),
    })
    const data = await res.json()
    if (data.code === 200) {
      message.success(t('login.codeSent'))
      codeSent.value = true
      startCooldown()
    } else {
      message.error(data.message || t('forgotPwd.networkFail'))
    }
  } catch {
    message.error(t('forgotPwd.networkFail'))
  } finally {
    sendingCode.value = false
  }
}

const startCooldown = () => {
  codeCooldown.value = 60
  clearInterval(cooldownTimer)
  cooldownTimer = setInterval(() => {
    codeCooldown.value--
    if (codeCooldown.value <= 0) clearInterval(cooldownTimer)
  }, 1000)
}

const handleSubmit = async () => {
  await formRef.value?.validate(async (valid) => {
    if (!valid) return
    loading.value = true
    try {
      const res = await fetch('/api/v1/auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: form.email,
          code: form.code,
          newPassword: form.newPassword,
        }),
      })
      const data = await res.json()
      if (data.code === 200) {
        message.success(t('forgotPwd.successMsg'))
        router.push('/login')
      } else {
        message.error(data.message || t('forgotPwd.networkFail'))
      }
    } catch {
      message.error(t('forgotPwd.networkFail'))
    } finally {
      loading.value = false
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
.auth-form { margin-bottom: 24px; }
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
}
.code-row {
  display: flex;
  gap: 10px;
  width: 100%;
}
.code-row .el-input { flex: 1; }
.send-code-btn {
  white-space: nowrap;
  min-width: 110px;
}
</style>
