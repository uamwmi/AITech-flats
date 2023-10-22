import './score.css';

function Score({ fraction, label, isMarked }) {
  return (
    <span
      className={`score-container main-font ${
        isMarked ? 'score-container-marked' : ''
      }`}
    >
      <div className="score-number-label">{`${Math.round(
        fraction * 100
      )}%`}</div>
      {` ${label}`}
    </span>
  );
}

export default Score;
