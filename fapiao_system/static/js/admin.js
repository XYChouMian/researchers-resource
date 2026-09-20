/* 后台管理页脚本：成员账号管理。 */
"use strict";

let me = null;
let users = [];

async function loadUsers() {
  users = (await api("api/admin/users")).users;
  const tbody = document.querySelector("#users-table tbody");
  tbody.innerHTML = users.map((u) => `
    <tr>
      <td>${esc(u.student_id)}</td>
      <td>${esc(u.name)}</td>
      <td>${u.role === "admin" ? "管理员" : "用户"}</td>
      <td class="num">${u.record_count}</td>
      <td class="num">${u.open_count}</td>
      <td class="actions">
        <button class="btn btn-sm" data-act="edit" data-id="${u.id}">编辑</button>
        <button class="btn btn-sm" data-act="reset" data-id="${u.id}">重置密码</button>
        <button class="btn btn-sm btn-danger" data-act="delete" data-id="${u.id}">删除</button>
      </td>
    </tr>`).join("");
}

document.querySelector("#users-table tbody").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const user = users.find((u) => u.id === Number(btn.dataset.id));
  try {
    if (btn.dataset.act === "edit") {
      openEditUserModal(user);
    } else if (btn.dataset.act === "reset") {
      if (!confirm(`确定将 ${user.student_id}（${user.name}）的密码重置为学号吗？`)) return;
      await api(`api/admin/users/${user.id}/reset_password`, { json: {} });
      toast("密码已重置为学号");
    } else if (btn.dataset.act === "delete") {
      if (!confirm(`确定删除用户 ${user.student_id}（${user.name}）吗？`)) return;
      await api(`api/admin/users/${user.id}`, { method: "DELETE" });
      toast("已删除");
      loadUsers();
    }
  } catch (err) {
    toast(err.message, false);
  }
});

function openEditUserModal(user) {
  const overlay = showModal(`编辑用户 - ${user.student_id}`, `
    <form id="user-edit-form" class="form-col">
      <div class="field">
        <span class="field-label">姓名</span>
        <input name="name" required maxlength="50" value="${esc(user.name)}">
      </div>
      <div class="field">
        <span class="field-label">角色</span>
        <select name="role">
          <option value="user">用户</option>
          <option value="admin">管理员</option>
        </select>
      </div>
      <button type="submit" class="btn btn-primary">保存</button>
    </form>`);
  const form = overlay.querySelector("#user-edit-form");
  form.role.value = user.role;
  if (user.id === me.id) form.role.disabled = true;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api(`api/admin/users/${user.id}`, {
        method: "PUT",
        json: { name: form.name.value.trim(), role: form.role.value },
      });
      closeModal();
      toast("已保存");
      loadUsers();
    } catch (err) {
      toast(err.message, false);
    }
  });
}

document.getElementById("user-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    const { user } = await api("api/admin/users", {
      json: {
        student_id: form.student_id.value.trim(),
        name: form.name.value.trim(),
        role: form.role.value,
      },
    });
    toast(`用户 ${user.student_id} 已创建，初始密码为学号`);
    form.reset();
    loadUsers();
  } catch (err) {
    toast(err.message, false);
  }
});

loadMe("admin").then((user) => {
  me = user;
  if (user) loadUsers();
});
