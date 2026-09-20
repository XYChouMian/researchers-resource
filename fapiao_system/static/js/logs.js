/* 系统日志页脚本：展示后台操作日志（操作筛选 + 分页）。 */
"use strict";

let entries = [];
const pager = { page: 1, pageSize: 20 };
const actionFilter = document.getElementById("action-filter");
actionFilter.addEventListener("change", () => {
  pager.page = 1;
  render();
});

async function loadLogs() {
  const data = await api("api/admin/logs");
  entries = data.entries;
  render();
}

function filteredEntries() {
  if (!actionFilter.value) return entries;
  return entries.filter((e) => e.action === actionFilter.value);
}

function render() {
  const shown = filteredEntries();
  document.getElementById("logs-empty").classList.toggle("hidden", shown.length > 0);
  const tbody = document.querySelector("#logs-table tbody");
  tbody.innerHTML = paginate(shown, pager).map((e) => {
    if (e.raw) {
      return `<tr><td colspan="9" class="muted">${esc(e.raw)}</td></tr>`;
    }
    return `<tr>
      <td>${esc(e.time)}</td>
      <td>${esc(e.action_text || e.action)}</td>
      <td>${esc(e.operator)}（${esc(e.operator_id)}）</td>
      <td>${esc(e.student_id)}</td>
      <td>${esc(e.name)}</td>
      <td>${esc(e.product_name)}</td>
      <td class="num">${fmtMoney(e.amount)}</td>
      <td>${esc(e.from_status)}→${esc(e.to_status)}</td>
      <td class="remark-cell">${esc(e.note || "")}</td>
    </tr>`;
  }).join("");
  renderPagination(document.getElementById("logs-pagination"), pager, shown.length, render);
}

loadMe("admin").then((user) => {
  if (user) loadLogs();
});
