/* 公共脚本：api() 封装、esc()、toast、弹窗、修改密码、顶栏公共事件。 */
"use strict";

const STATUS_TEXT = { pending: "待开票", invoiced: "已开票", reimbursed: "已报销", rejected: "已驳回", processed: "已处理" };
const STATUS_CLASS = { pending: "st-pending", invoiced: "st-invoiced", reimbursed: "st-reimbursed", rejected: "st-rejected", processed: "st-processed" };
const CHANNEL_TEXT = { taobao: "淘宝", jd: "京东", other: "其他" };
const PAYER_TEXT = { self: "本人", tang: "唐老师" };
const FILE_TYPE_TEXT = { invoice: "发票", attachment: "附件" };

async function api(path, options = {}) {
  const opts = { ...options };
  if (opts.json !== undefined) {
    opts.method = opts.method || "POST";
    opts.headers = { "Content-Type": "application/json", ...opts.headers };
    opts.body = JSON.stringify(opts.json);
    delete opts.json;
  }
  const res = await fetch(path, opts);
  if (res.status === 401 && !path.startsWith("api/login")) {
    location.href = "./";
    throw new Error("未登录");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `请求失败（${res.status}）`);
  return data;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function toast(message, ok = true) {
  const el = document.createElement("div");
  el.className = `toast ${ok ? "toast-ok" : "toast-err"}`;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function todayLocal() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function showModal(title, bodyHtml) {
  closeModal();
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.id = "modal-root";
  overlay.innerHTML = `
    <div class="modal card">
      <h3>${esc(title)}</h3>
      <div class="modal-body">${bodyHtml}</div>
    </div>`;
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeModal();
  });
  document.body.appendChild(overlay);
  return overlay;
}

function closeModal() {
  const el = document.getElementById("modal-root");
  if (el) el.remove();
}

function filePreviewUrl(id) {
  return `api/invoices/${id}/download?inline=1`;
}

function shortName(name, max = 12) {
  return name.length > max ? name.slice(0, 9) + "…" : name;
}

function fmtMoney(amount) {
  return Number(amount).toFixed(2);
}

async function downloadFile(url, fallbackName) {
  const res = await fetch(url);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `下载失败（${res.status}）`);
  }
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") || "";
  const match = /filename\*=UTF-8''([^;]+)/.exec(disposition);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = match ? decodeURIComponent(match[1]) : fallbackName;
  link.click();
  URL.revokeObjectURL(link.href);
}

const ACTION_TEXT = {
  reimburse: "标记已报销", reject: "驳回", revert: "恢复未报销", process: "标记已处理",
};

async function openRecordDetail(rid) {
  const { record, history } = await api(`api/records/${rid}/detail`);
  const infoRows = [
    { label: "用户备注", value: record.remark },
    { label: "报销备注", value: record.reimburse_note },
    { label: "驳回理由", value: record.reject_note },
    { label: "处理说明", value: record.process_note },
  ].filter((row) => row.value);
  const overlay = showModal(`详情 - ${record.product_name}`, `
    ${infoRows.map((row) => `<p class="detail-info"><b>${row.label}：</b>${esc(row.value)}</p>`).join("")}
    <h3>发票和附件</h3>
    <div class="table-scroll">
      <table class="data-table upload-table">
        <thead><tr><th>文件名（点击预览）</th><th>类型</th><th>备注</th></tr></thead>
        <tbody>
          ${record.invoices.map((inv) => `
            <tr>
              <td><a href="${filePreviewUrl(inv.id)}" target="_blank" rel="noopener" title="${esc(inv.orig_name)}">${esc(inv.orig_name)}</a></td>
              <td><span class="badge ${inv.category === "invoice" ? "st-invoiced" : "st-pending"}">${FILE_TYPE_TEXT[inv.category] ?? "附件"}</span></td>
              <td class="remark-cell">${esc(inv.remark || "")}</td>
            </tr>`).join("") || '<tr><td colspan="3" class="muted">未上传</td></tr>'}
        </tbody>
      </table>
    </div>
    <h3>操作历史</h3>
    <div class="table-scroll">
      <table class="data-table upload-table">
        <thead><tr><th>时间</th><th>操作</th><th>经办人</th><th>备注</th></tr></thead>
        <tbody>
          ${history.map((h) => `
            <tr>
              <td>${esc(h.created_at)}</td>
              <td>${ACTION_TEXT[h.action] ?? h.action}</td>
              <td>${esc(h.operator)}</td>
              <td>${esc(h.note || "")}</td>
            </tr>`).join("") || '<tr><td colspan="4" class="muted">暂无记录</td></tr>'}
        </tbody>
      </table>
    </div>`);
  overlay.querySelector(".modal").classList.add("modal-wide");
}

function paginate(list, pager) {
  const start = (pager.page - 1) * pager.pageSize;
  return list.slice(start, start + pager.pageSize);
}

function renderPagination(el, pager, total, onPageChange, selectedCount = 0) {
  const totalPages = Math.max(1, Math.ceil(total / pager.pageSize));
  pager.page = Math.min(Math.max(1, pager.page), totalPages);
  const countText = selectedCount > 0
    ? `已选 ${selectedCount} 条/共 ${total} 条`
    : `共 ${total} 条`;
  el.innerHTML = `
    <span class="muted">${countText}</span>
    <label class="page-size-label">每页
      <select class="page-size">
        ${[10, 20, 50].map((n) => `<option value="${n}" ${n === pager.pageSize ? "selected" : ""}>${n}</option>`).join("")}
      </select> 条
    </label>
    <button type="button" class="btn btn-sm" data-nav="prev" ${pager.page <= 1 ? "disabled" : ""}>‹ 上一页</button>
    <span>第 ${pager.page} / ${totalPages} 页</span>
    <button type="button" class="btn btn-sm" data-nav="next" ${pager.page >= totalPages ? "disabled" : ""}>下一页 ›</button>`;
  el.querySelector(".page-size").value = String(pager.pageSize);
  el.querySelector(".page-size").addEventListener("change", (e) => {
    pager.pageSize = Number(e.target.value);
    pager.page = 1;
    onPageChange();
  });
  el.querySelector('[data-nav="prev"]').addEventListener("click", () => {
    if (pager.page > 1) {
      pager.page -= 1;
      onPageChange();
    }
  });
  el.querySelector('[data-nav="next"]').addEventListener("click", () => {
    if (pager.page < totalPages) {
      pager.page += 1;
      onPageChange();
    }
  });
}

function showPasswordDialog() {
  const overlay = showModal("修改密码", `
    <form id="password-form" class="form-col">
      <div class="field">
        <span class="field-label">原密码</span>
        <input name="old_password" type="password" required>
      </div>
      <div class="field">
        <span class="field-label">新密码（至少 6 位）</span>
        <input name="new_password" type="password" required minlength="6">
      </div>
      <div class="field">
        <span class="field-label">确认新密码</span>
        <input name="confirm_password" type="password" required minlength="6">
      </div>
      <p class="form-error" id="password-error"></p>
      <button type="submit" class="btn btn-primary">保存</button>
    </form>`);
  overlay.querySelector("#password-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const errEl = overlay.querySelector("#password-error");
    if (fd.get("new_password") !== fd.get("confirm_password")) {
      errEl.textContent = "两次输入的新密码不一致";
      return;
    }
    try {
      await api("api/password", {
        json: { old_password: fd.get("old_password"), new_password: fd.get("new_password") },
      });
      closeModal();
      toast("密码修改成功");
    } catch (err) {
      errEl.textContent = err.message;
    }
  });
}

async function loadMe(expectedRole) {
  const user = (await api("api/me")).user;
  if (expectedRole && user.role !== expectedRole) {
    location.href = user.role === "admin" ? "reimburse" : "user";
    return null;
  }
  document.body.classList.toggle("is-admin", user.role === "admin");
  const info = document.getElementById("me-info");
  if (info) {
    info.textContent = `${user.name}（${user.role === "admin" ? "管理员" : "用户"}）`;
  }
  return user;
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btn-logout")?.addEventListener("click", async () => {
    await api("api/logout", { json: {} }).catch(() => {});
    location.href = "./";
  });
  document.getElementById("btn-pass")?.addEventListener("click", showPasswordDialog);
});
