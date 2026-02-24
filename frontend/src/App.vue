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
            <router-link to="/login">
              <el-button type="primary" round size="small">{{ $t('app.login') }}</el-button>
            </router-link>
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
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { Monitor, CopyDocument } from '@element-plus/icons-vue'

const { t, locale } = useI18n()

const currentLang = computed(() => locale.value)
const toggleLanguage = () => {
  locale.value = locale.value === 'zh' ? 'en' : 'zh'
}
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
