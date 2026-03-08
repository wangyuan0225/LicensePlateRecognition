<template>
  <div class="app-layout">
    <el-config-provider :builtin-theme="'light'">
      <header class="app-header">
        <div class="header-content">
          <div class="logo" @click="$router.push('/')">
            <el-icon :size="24" color="var(--primary-color)">
              <Monitor />
            </el-icon>
            <span class="logo-text">{{ $t('app.title') }}</span>
          </div>
          <nav class="nav-links">
            <router-link to="/analyze" class="nav-item">{{ $t('app.analyze') }}</router-link>
            <router-link to="/history" class="nav-item">{{ $t('app.history') }}</router-link>
            <div class="divider"></div>
            <el-button link @click="toggleLanguage" class="lang-switcher">
              <el-icon>
                <CopyDocument />
              </el-icon>
              {{ currentLang === 'zh' ? 'EN' : '中文' }}
            </el-button>

            <!-- Not logged in: show login button -->
            <template v-if="!isLoggedIn">
              <router-link to="/login">
                <el-button type="primary" round size="small">{{ $t('app.login') }}</el-button>
              </router-link>
            </template>

            <!-- Logged in: show username + logout -->
            <template v-else>
              <el-dropdown @command="handleUserCommand">
                <span class="user-info">
                  <el-icon><User /></el-icon>
                  <span class="username-text">{{ username }}</span>
                  <el-icon class="el-icon--right"><ArrowDown /></el-icon>
                </span>
                <template #dropdown>
                  <el-dropdown-menu>
                    <el-dropdown-item command="changePassword">
                      <el-icon><Lock /></el-icon>
                      {{ $t('app.changePassword') }}
                    </el-dropdown-item>
                    <el-dropdown-item command="logout" divided>
                      <el-icon><SwitchButton /></el-icon>
                      {{ $t('app.logout') }}
                    </el-dropdown-item>
                  </el-dropdown-menu>
                </template>
              </el-dropdown>
            </template>
          </nav>
        </div>
      </header>

      <main class="app-main">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </main>

      <footer class="app-footer">
        <p>{{ $t('app.footer') }}</p>
      </footer>
    </el-config-provider>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import { Monitor, CopyDocument, User, ArrowDown, SwitchButton, Lock } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useMessage } from '@/composables/useMessage'

const { t, locale } = useI18n()
const router = useRouter()
const route = useRoute()
const message = useMessage()

const currentLang = computed(() => locale.value)
const toggleLanguage = () => {
  locale.value = locale.value === 'zh' ? 'en' : 'zh'
}

const isLoggedIn = ref(false)
const username = ref('')

const checkLoginStatus = () => {
  const token = localStorage.getItem('token')
  const userStr = localStorage.getItem('user')
  if (token && userStr) {
    try {
      const user = JSON.parse(userStr)
      isLoggedIn.value = true
      username.value = user.username || user.email || 'User'
    } catch {
      isLoggedIn.value = false
      username.value = ''
    }
  } else {
    isLoggedIn.value = false
    username.value = ''
  }
}

const handleUserCommand = (command) => {
  if (command === 'changePassword') {
    router.push('/change-password')
  } else if (command === 'logout') {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    isLoggedIn.value = false
    username.value = ''
    message.success(t('app.logoutSuccess'))
    router.push('/')
  }
}

// Check on mount and on route change (to catch login redirect)
onMounted(checkLoginStatus)
watch(() => route.path, checkLoginStatus)
</script>

<style scoped>
.app-layout {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-color);
  height: 60px;
  display: flex;
  align-items: center;
}

.header-content {
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  transition: opacity 0.2s;
}

.logo:hover {
  opacity: 0.8;
}

.logo-text {
  font-weight: 600;
  font-size: 1.1rem;
  letter-spacing: -0.02em;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 24px;
}

.nav-item {
  text-decoration: none;
  color: var(--text-secondary);
  font-size: 0.95rem;
  font-weight: 500;
  transition: color 0.2s;
}

.nav-item:hover,
.nav-item.router-link-active {
  color: var(--text-primary);
}

.lang-switcher {
  color: var(--text-secondary);
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 4px;
}

.lang-switcher:hover {
  color: var(--text-primary);
}

.divider {
  width: 1px;
  height: 16px;
  background-color: var(--border-color);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  color: var(--text-primary);
  font-weight: 500;
  font-size: 0.95rem;
  padding: 6px 12px;
  border-radius: 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  transition: all 0.2s;
}

.user-info:hover {
  border-color: var(--primary-color);
  color: var(--primary-color);
}

.username-text {
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.app-main {
  flex: 1;
  background-color: var(--bg-color);
}

.app-footer {
  text-align: center;
  padding: 32px;
  color: var(--text-secondary);
  font-size: 0.85rem;
  border-top: 1px solid var(--border-color);
}

/* Page transition */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(10px);
}
</style>

<style>
/* ============================================================
   全局强制覆盖 ElMessage 样式 — 白底 + 黑色边框
   ============================================================ */
div.el-message.lpr-message {
  background-color: #ffffff !important;
  border: 1.5px solid #1a1a1a !important;
  border-radius: 10px !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10) !important;
  padding: 12px 20px !important;
  width: fit-content !important;
  max-width: 90vw !important;
}

div.el-message.lpr-message .el-message__content,
div.el-message.lpr-message .el-message__icon {
  font-weight: 500 !important;
  font-size: 0.95rem !important;
}

/* warning → 红字 */
div.el-message.lpr-message.el-message--warning .el-message__content,
div.el-message.lpr-message.el-message--warning .el-message__icon {
  color: #dc2626 !important;
}

/* success → 绿字 */
div.el-message.lpr-message.el-message--success .el-message__content,
div.el-message.lpr-message.el-message--success .el-message__icon {
  color: #16a34a !important;
}

/* error → 深红字 */
div.el-message.lpr-message.el-message--error .el-message__content,
div.el-message.lpr-message.el-message--error .el-message__icon {
  color: #b91c1c !important;
}

/* info → 深灰字 */
div.el-message.lpr-message.el-message--info .el-message__content,
div.el-message.lpr-message.el-message--info .el-message__icon {
  color: #374151 !important;
}

/* 关闭按钮统一黑色 */
div.el-message.lpr-message .el-message__closeBtn {
  color: #6b7280 !important;
}
div.el-message.lpr-message .el-message__closeBtn:hover {
  color: #111827 !important;
}
</style>
