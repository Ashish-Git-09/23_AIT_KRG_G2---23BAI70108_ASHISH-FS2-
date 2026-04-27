import React, { useState } from "react";
import items from "../data/items";

const Pagination = () => {
  const itemsPerPage = 5;
  const totalPages = Math.ceil(items.length / itemsPerPage);
  const [currentPage, setCurrentPage] = useState(1);

  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentItems = items.slice(
    startIndex,
    startIndex + itemsPerPage
  );

  return (
    <div className="glass-container">
      <h2 className="heading">✨ Paginated Items</h2>

      <ul className="glass-list">
        {currentItems.map((item) => (
          <li key={item.id} className="glass-item">
            {item.name}
          </li>
        ))}
      </ul>

      <div className="controls">
        <button
          className="glass-btn"
          onClick={() => setCurrentPage(currentPage - 1)}
          disabled={currentPage === 1}
        >
          ← Prev
        </button>

        <span className="page-badge">
          {currentPage} / {totalPages}
        </span>

        <button
          className="glass-btn"
          onClick={() => setCurrentPage(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          Next →
        </button>
      </div>
    </div>
  );
};

export default Pagination;