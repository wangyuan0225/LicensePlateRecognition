<template>
  <div class="auth-container page-container">
    <div class="auth-card notion-card">
      <div class="auth-header">
        <h2 class="auth-title">{{ $t('changePwd.title') }}</h2>
        <p class="auth-subtitle">{{ $t('changePwd.subtitle') }}</p>
        <p class="email-badge">{{ userEmail }}</p>
      </div>

      <el-form ref="formRef" :model="form" :rules="rules" label-position="top" class="auth-form">
        <!-- 验证码 -->
        <el-form-item :label="$t('changePwd.verifyCode')" prop="code">
          <div class="code-row">
            <el-input v-model="form.code" :placeholder="$t('changePwd.verifyCodePlaceholder')"
              prefix-icon="Key" maxlength="6" />
            <el-button
              type="primary" plain
              :disabled="codeCooldown > 0"
              :loading="sendingCode"
              @click="sendCode"
              class="send-code-btn">
              {{ codeCooldown > 0 ? `${codeCooldown}s` : (codeSent ? $t('changePwd.sendCodeAgain') : $t('changePwd.sendCode')) }}
            </el-button>
          </div>
        </el-form-item>

        <!-- 旧密码 -->
        <el-form-item :label="$t('changePwd.oldPassword')" prop="oldPassword">
          <el-input v-model="form.oldPassword" type="password"
            :placeholder="$t('changePwd.oldPasswordPlaceholder')" prefix-icon="Lock" show-password />
        </el-form-item>

        <!-- 新密码 -->
        <el-form-item :label="$t('changePwd.newPassword')" prop="newPassword">
          <el-input v-model="form.newPassword" type="password"
            :placeholder="$t('changePwd.newPasswordPlaceholder')" prefix-icon="Lock" show-password />
        </el-form-item>

        <el-button type="primary" class="submit-btn" :loading="loading" @click="handleSubmit">
          {{ $t('changePwd.submitBtn') }}
        </el-button>
      </el-form>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
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

const userEmail = ref('')

onMounted(() => {
  const user = JSON.parse(localStorage.getItem('user') || '{}')
  userEmail.value = user.email || ''
})

const form = reactive({ code: '', oldPassword: '', newPassword: '' })

const rules = computed(() => ({
  code: [
    { required: true, message: t('changePwd.ruleCodeRequired'), trigger: 'blur' },
    { len: 6, message: t('changePwd.ruleCodeLength'), trigger: 'blur' },
  ],
  oldPassword: [
    { required: true, message: t('changePwd.ruleOldPasswordRequired'), trigger: 'blur' },
  ],
  newPassword: [
    { required: true, message: t('changePwd.ruleNewPasswordRequired'), trigger: 'blur' },
    { min: 6, max: 20, message: t('changePwd.rulePasswordLength'), trigger: 'blur' },
  ],
}))

const sendCode = async () => {
  if (!userEmail.value) {
    message.error('未能获取用户邮箱，请重新登录')
    return
  }
  sendingCode.value = true
  try {
    const res = await fetch('/api/v1/auth/send-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: userEmail.value, type: 'change' }),
    })
    const data = await res.json()
    if (data.code === 200) {
      message.success(t('login.codeSent'))
      codeSent.value = true
      startCooldown()
    } else {
      message.error(data.message || t('changePwd.networkFail'))
    }
  } catch {
    message.error(t('changePwd.networkFail'))
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
    const token = localStorage.getItem('token')
    try {
      const res = await fetch('/api/v1/auth/change-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          code: form.code,
          oldPassword: form.oldPassword,
          newPassword: form.newPassword,
        }),
      })
      const data = await res.json()
      if (data.code === 200) {
        message.success(t('changePwd.successMsg'))
        localStorage.removeItem('token')
        localStorage.removeItem('user')
        router.push('/login')
      } else {
        message.error(data.message || t('changePwd.networkFail'))
      }
    } catch {
      message.error(t('changePwd.networkFail'))
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
  margin-bottom: 8px;
}
.email-badge {
  display: inline-block;
  background: var(--bg-secondary, #f3f4f6);
  color: var(--primary-color, #2563eb);
  font-size: 0.85rem;
  font-weight: 500;
  padding: 4px 12px;
  border-radius: 20px;
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
