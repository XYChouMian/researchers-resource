/* 公共脚本：api() 封装、esc()、toast、弹窗、修改密码、顶栏公共事件。 */
"use strict";

const STATUS_TEXT = { pending: "待开票", invoiced: "已开票", reimbursed: "已报销" };
const STATUS_CLASS = { pending: "st-pending", invoiced: "st-invoiced", reimbursed: "st-reimbursed" };
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
    location.href = user.role === "admin" ? "admin" : "user";
    return null;
  }
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
