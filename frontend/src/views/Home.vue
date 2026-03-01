<template>
  <div class="home-container">
    <section class="hero-section">
      <div class="hero-content">
        <h1 class="hero-title">{{ $t('home.heroTitle') }}</h1>
        <p class="hero-subtitle">
          {{ $t('home.heroSubtitle') }}
        </p>
        <div class="hero-actions">
          <el-button type="primary" size="large" @click="goToAnalyze" class="action-btn">
            {{ $t('home.startAnalysis') }}
            <el-icon class="el-icon--right">
              <Right />
            </el-icon>
          </el-button>
          <el-button size="large" @click="goToHistory" class="action-btn secondary">
            {{ $t('home.viewHistory') }}
          </el-button>
        </div>
      </div>
      <div class="hero-image-placeholder notion-card">
        <!-- Visual placeholder for the system -->
        <div class="demo-ui">
          <div class="demo-header">{{ $t('home.demoHeader') }}</div>
          <div class="demo-box">
            <img src="@/assets/demo-result.png" alt="Demo Recognition Result" class="demo-image" />
          </div>
        </div>
      </div>
    </section>

    <section class="features-section">
      <div class="feature-grid">
        <div class="feature-card notion-card" v-for="(feature, index) in features" :key="index">
          <el-icon class="feature-icon" :size="32">
            <component :is="feature.icon" />
          </el-icon>
          <h3 class="feature-title">{{ feature.title }}</h3>
          <p class="feature-desc">{{ feature.description }}</p>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { Right, Aim, DataLine, Picture } from '@element-plus/icons-vue'

const router = useRouter()
const { t } = useI18n()

const goToAnalyze = () => {
  router.push('/analyze')
}

const goToHistory = () => {
  router.push('/history')
}

const features = computed(() => [
  {
    icon: 'Aim',
    title: t('home.feature1Title'),
    description: t('home.feature1Desc')
  },
  {
    icon: 'DataLine',
    title: t('home.feature2Title'),
    description: t('home.feature2Desc')
  },
  {
    icon: 'Picture',
    title: t('home.feature3Title'),
    description: t('home.feature3Desc')
  }
])
</script>

<style scoped>
.home-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 60px 24px;
}

.hero-section {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 60px;
  margin-bottom: 100px;
  min-height: 50vh;
}

.hero-content {
  flex: 1;
  max-width: 560px;
}

.hero-title {
  font-size: 3.5rem;
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: -0.03em;
  margin-bottom: 24px;
  color: var(--text-primary);
}

.hero-subtitle {
  font-size: 1.2rem;
  line-height: 1.6;
  color: var(--text-secondary);
  margin-bottom: 40px;
}

.hero-actions {
  display: flex;
  gap: 16px;
}

.action-btn {
  font-size: 1.05rem;
  padding: 12px 24px;
  height: auto;
  border-radius: 8px;
}

.secondary {
  border-color: var(--border-color);
  color: var(--text-primary);
}

.secondary:hover {
  background-color: var(--bg-secondary);
  border-color: var(--border-color);
}

.hero-image-placeholder {
  flex: 1;
  height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-secondary);
  padding: 0;
  overflow: hidden;
}

.demo-ui {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.demo-header {
  padding: 16px;
  border-bottom: 1px solid var(--border-color);
  font-weight: 500;
  font-size: 0.9rem;
  color: var(--text-secondary);
  background: white;
}

.demo-box {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #1a1a2e;
  position: relative;
  overflow: hidden;
}

.demo-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.features-section {
  padding-top: 40px;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 32px;
}

.feature-card {
  padding: 32px;
}

.feature-icon {
  color: var(--accent-color);
  margin-bottom: 20px;
}

.feature-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 12px;
  color: var(--text-primary);
}

.feature-desc {
  color: var(--text-secondary);
  line-height: 1.5;
  font-size: 0.95rem;
}

@media (max-width: 900px) {
  .hero-section {
    flex-direction: column;
    text-align: center;
  }

  .hero-content {
    margin: 0 auto;
  }

  .hero-actions {
    justify-content: center;
  }

  .hero-image-placeholder {
    width: 100%;
  }
}
</style>
