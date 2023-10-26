import './loading.css';

function Loading() {
  return (
    <div className="loading-container">
      <span className="loading-spinner" />
      <p className="loading-label main-font filling-layout-container">
        Loading...
      </p>
    </div>
  );
}

export default Loading;
