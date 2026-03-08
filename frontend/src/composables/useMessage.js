/**
 * useMessage — 统一封装 ElMessage，自动附加 lpr-message 自定义样式类
 *
 * 用法：与 ElMessage 完全相同，但无需手动指定 customClass
 *   import { useMessage } from '@/composables/useMessage'
 *   const message = useMessage()
 *   message.success('操作成功')
 *   message.warning('请先登录')
 *   message.error('请求失败')
 */
import { ElMessage } from 'element-plus'

const DURATION = 2500

function show(type, options) {
    const opts = typeof options === 'string' ? { message: options } : { ...options }
    return ElMessage({
        duration: DURATION,
        showClose: false,
        ...opts,
        type,
        customClass: `lpr-message lpr-message--${type}${opts.customClass ? ' ' + opts.customClass : ''}`,
    })
}

export function useMessage() {
    return {
        success: (opts) => show('success', opts),
        warning: (opts) => show('warning', opts),
        error: (opts) => show('error', opts),
        info: (opts) => show('info', opts),
    }
}
