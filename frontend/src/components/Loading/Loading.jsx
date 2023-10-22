import './loading.css';

function Loading() {
  return (
    <div className="loading-container">
      <span className="loading-spinner" />
      <p className="loading-label main-font filling-layout-container">
        Ładowanie
      </p>
    </div>
  );
}

export default Loading;
