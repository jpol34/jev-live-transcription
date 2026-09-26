function titleCase(slug) {
  return slug
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

const CATEGORY_LABELS = {
  prospect: "Prospect calls",
  resident: "Resident calls",
};

async function loadCalls() {
  const container = document.getElementById("calls");
  let grouped;
  try {
    const response = await fetch("/calls");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    grouped = await response.json();
  } catch (err) {
    container.textContent = `Failed to load calls: ${err}`;
    return;
  }

  container.innerHTML = "";
  for (const category of Object.keys(grouped).sort()) {
    const categorySection = document.createElement("section");
    categorySection.className = "category-group";

    const heading = document.createElement("h2");
    heading.textContent = CATEGORY_LABELS[category] || titleCase(category);
    categorySection.appendChild(heading);

    const subtypes = grouped[category];
    for (const subtype of Object.keys(subtypes).sort()) {
      const subtypeSection = document.createElement("div");
      subtypeSection.className = "subtype-group";

      const subtypeHeading = document.createElement("h3");
      subtypeHeading.textContent = titleCase(subtype);
      subtypeSection.appendChild(subtypeHeading);

      const grid = document.createElement("div");
      grid.className = "call-grid";
      for (const call of subtypes[subtype]) {
        const link = document.createElement("a");
        link.className = "call-chip";
        link.href = `/call/${call.call_id}`;
        link.innerHTML = `Call ${call.call_id} &middot; ~${call.target_minutes}m`;
        if (call.edge_case) {
          const flag = document.createElement("span");
          flag.className = "edge-flag";
          flag.textContent = "edge case";
          link.appendChild(flag);
        }
        grid.appendChild(link);
      }
      subtypeSection.appendChild(grid);
      categorySection.appendChild(subtypeSection);
    }
    container.appendChild(categorySection);
  }
}

loadCalls();
