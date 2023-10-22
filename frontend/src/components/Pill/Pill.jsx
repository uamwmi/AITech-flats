import './pill.css';

function Pill({ selected, children, className }) {
  return (
    <div
      className={`filling-layout-container pill-container ${
        selected ? 'pill-selected' : ''
      } ${className || ''}`}
    >
      {children}
    </div>
  );
}

export default Pill;
