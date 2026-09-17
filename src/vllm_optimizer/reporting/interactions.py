"""Local-only controls for a standalone report, with keyboard-accessible buttons."""


def report_script() -> str:
    return """<script>
document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    const status = button.nextElementSibling?.matches('.copy-status')
      ? button.nextElementSibling : target.nextElementSibling;
    try {
      await navigator.clipboard.writeText(target.textContent);
      status.textContent = 'Copied';
    } catch {
      let parent = target.parentElement;
      while (parent) { if (parent.tagName === 'DETAILS') parent.open = true; parent = parent.parentElement; }
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target); selection.removeAllRanges(); selection.addRange(range);
      status.textContent = 'Text selected. Press Ctrl+C or Cmd+C to copy.';
    }
  });
});
document.querySelectorAll('nav a').forEach(link => {
  link.addEventListener('click', () => {
    let target = document.querySelector(link.getAttribute('href'));
    while (target) { if (target.tagName === 'DETAILS') target.open = true; target = target.parentElement; }
  });
});
document.querySelectorAll('[data-sort]').forEach(button => {
  button.addEventListener('click', () => {
    const table = button.closest('table');
    const column = Number(button.dataset.sort);
    const ascending = button.closest('th').getAttribute('aria-sort') !== 'ascending';
    table.querySelectorAll('th').forEach(th => th.removeAttribute('aria-sort'));
    button.closest('th').setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
    const rows = Array.from(table.tBodies[0].rows);
    rows.sort((a, b) => {
      const x = a.cells[column].dataset.value, y = b.cells[column].dataset.value;
      if (x === '') return y === '' ? 0 : 1;
      if (y === '') return -1;
      const difference = button.dataset.type === 'number' ? Number(x) - Number(y) : x.localeCompare(y);
      return ascending ? difference : -difference;
    });
    rows.forEach(row => table.tBodies[0].appendChild(row));
  });
});
</script>"""
