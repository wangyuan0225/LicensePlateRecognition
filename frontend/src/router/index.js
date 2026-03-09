import { createRouter, createWebHistory } from 'vue-router'
import { ElMessage } from 'element-plus'
import i18n from '../i18n'

const routes = [
    {
        path: '/',
        name: 'Home',
        component: () => import('../views/Home.vue')
    },
    {
        path: '/login',
        name: 'Login',
        component: () => import('../views/Login.vue')
    },
    {
        path: '/analyze',
        name: 'Analyze',
        component: () => import('../views/Analyze.vue'),
        meta: { requiresAuth: true }
    },
    {
        path: '/history',
        name: 'History',
        component: () => import('../views/History.vue'),
        meta: { requiresAuth: true }
    },
    {
        path: '/feedback',
        name: 'Feedback',
        component: () => import('../views/Feedback.vue'),
        meta: { requiresAuth: true }
    },
    {
        path: '/forgot-password',
        name: 'ForgotPassword',
        component: () => import('../views/ForgotPassword.vue')
    },
    {
        path: '/change-password',
        name: 'ChangePassword',
        component: () => import('../views/ChangePassword.vue'),
        meta: { requiresAuth: true }
    },
    {
        path: '/force-change-password',
        name: 'ForceChangePassword',
        component: () => import('../views/ForceChangePassword.vue'),
        meta: { requiresAuth: true, hideLayout: true }
    },
    {
        path: '/admin/history',
        name: 'AdminHistory',
        component: () => import('../views/AdminHistory.vue'),
        meta: { requiresAuth: true, role: 'ADMIN' }
    },
    {
        path: '/admin/feedback',
        name: 'AdminFeedback',
        component: () => import('../views/AdminFeedback.vue'),
        meta: { requiresAuth: true, role: 'ADMIN' }
    }
]

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes
})

// Navigation guard: 未登录时显示提示并跳转到登录页，以及强制修改密码拦截
router.beforeEach((to, from, next) => {
    const token = localStorage.getItem('token')
    const userStr = localStorage.getItem('user')
    let user = null
    if (userStr) {
        try { user = JSON.parse(userStr) } catch (e) { }
    }

    // 1. 强制修改密码拦截: 如果需要改密码，且目标并非强改页面/登出等
    if (user && user.forceChangePassword && to.path !== '/force-change-password' && to.path !== '/login') {
        next('/force-change-password')
        return
    }

    // 2. 权限校验
    if (to.meta.requiresAuth) {
        if (!token) {
            // 读取当前语言环境的提示文案
            const t = i18n.global.t
            ElMessage({
                message: t('app.notLoggedIn'),
                type: 'warning',
                customClass: 'lpr-message lpr-message--warning',
                duration: 2500,
                showClose: false,
            })
            next({ name: 'Login', query: { redirect: to.fullPath } })
            return
        }

        // 3. 角色校验
        if (to.meta.role && user && user.role !== to.meta.role) {
            ElMessage.error('无权限访问')
            next('/')
            return
        }
    }

    next()
})

export default router
