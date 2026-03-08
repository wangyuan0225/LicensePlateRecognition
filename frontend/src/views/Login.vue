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

        <!-- 注册：用户名 -->
        <transition name="el-fade-in-linear">
          <el-form-item v-if="!isLogin" :label="$t('login.username')" prop="username">
            <el-input v-model="form.username" :placeholder="$t('login.usernamePlaceholder')" prefix-icon="User" />
          </el-form-item>
        </transition>

        <!-- 登录：账号（用户名或邮箱） -->
        <el-form-item v-if="isLogin" :label="$t('login.identifier')" prop="identifier">
          <el-input
            v-model="form.identifier"
            :placeholder="$t('login.identifierPlaceholder')"
            prefix-icon="Message" />
        </el-form-item>

        <!-- 注册：邮箱 -->
        <el-form-item v-else :label="$t('login.email')" prop="email">
          <el-input
            v-model="form.email"
            :placeholder="$t('login.emailPlaceholder')"
            prefix-icon="Message" />
        </el-form-item>

        <!-- 注册：验证码 -->
        <transition name="el-fade-in-linear">
          <el-form-item v-if="!isLogin" :label="$t('login.verifyCode')" prop="code">
            <div class="code-row">
              <el-input v-model="form.code" :placeholder="$t('login.verifyCodePlaceholder')" prefix-icon="Key"
                maxlength="6" />
              <el-button
                type="primary"
                plain
                :disabled="codeCooldown > 0 || !form.email"
                :loading="sendingCode"
                @click="sendCode"
                class="send-code-btn">
                {{ codeCooldown > 0 ? `${codeCooldown}s` : (codeSent ? $t('login.sendCodeAgain') : $t('login.sendCode')) }}
              </el-button>
            </div>
          </el-form-item>
        </transition>

        <!-- 密码 -->
        <el-form-item :label="$t('login.password')" prop="password">
          <el-input v-model="form.password" type="password" :placeholder="$t('login.passwordPlaceholder')"
            prefix-icon="Lock" show-password />
        </el-form-item>

        <!-- 登录选项 -->
        <el-form-item v-if="isLogin">
          <div class="form-options">
            <el-checkbox v-model="form.rememberMe">{{ $t('login.rememberMe') }}</el-checkbox>
            <el-link type="primary" :underline="false" @click="$router.push('/forgot-password')">
              {{ $t('login.forgotPassword') }}
            </el-link>
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
const sendingCode = ref(false)
const codeSent = ref(false)
const codeCooldown = ref(0)
let cooldownTimer = null

const form = reactive({
  identifier: '',   // 登录用（用户名或邮箱）
  username: '',     // 注册用
  email: '',        // 注册用
  password: '',
  code: '',         // 注册验证码
  rememberMe: false
})

const rules = computed(() => ({
  identifier: [
    { required: true, message: t('login.ruleIdentifierRequired'), trigger: 'blur' },
  ],
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
  ],
  code: [
    { required: true, message: t('login.ruleCodeRequired'), trigger: 'blur' },
    { len: 6, message: t('login.ruleCodeLength'), trigger: 'blur' },
  ],
}))

const toggleMode = () => {
  isLogin.value = !isLogin.value
  formRef.value?.resetFields()
  codeSent.value = false
  codeCooldown.value = 0
  clearInterval(cooldownTimer)
}

/** 发送注册验证码 */
const sendCode = async () => {
  if (!form.email) return
  sendingCode.value = true
  try {
    const res = await fetch('/api/v1/auth/send-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: form.email, type: 'register' }),
    })
    const data = await res.json()
    if (data.code === 200) {
      ElMessage.success(t('login.codeSent'))
      codeSent.value = true
      startCooldown()
    } else {
      ElMessage.error(data.message || t('login.networkFail'))
    }
  } catch {
    ElMessage.error(t('login.networkFail'))
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
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (!valid) return
    loading.value = true

    try {
      if (isLogin.value) {
        // 登录
        const res = await fetch('/api/v1/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            identifier: form.identifier,
            password: form.password,
          }),
        })
        const data = await res.json()

        if (data.code === 200) {
          localStorage.setItem('token', data.data.token)
          localStorage.setItem('user', JSON.stringify(data.data.user))
          ElMessage.success(t('login.loginSuccess'))
          const redirect = router.currentRoute.value.query.redirect
          router.push(redirect || '/analyze')
        } else {
          ElMessage.error(data.message || t('login.loginFail'))
        }
      } else {
        // 注册
        const res = await fetch('/api/v1/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            username: form.username,
            email: form.email,
            password: form.password,
            code: form.code,
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
    } catch {
      ElMessage.error(t('login.networkFail'))
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

.code-row {
  display: flex;
  gap: 10px;
  width: 100%;
}

.code-row .el-input {
  flex: 1;
}

.send-code-btn {
  white-space: nowrap;
  min-width: 110px;
}
</style>
