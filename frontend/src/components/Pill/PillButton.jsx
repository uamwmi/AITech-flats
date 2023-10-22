import Pill from './Pill';

function PillButton({ className, children, selected, inactive, onClick }) {
  return (
    <button
      type="button"
      className={`${
        className ?? ''
      } pill-button-wrapper filling-layout-container`}
      onClick={(event) => (inactive ? null : onClick(event))}
    >
      <Pill
        className={`${
          inactive ? 'pill-button-state-inactive' : 'pill-button-interactable'
        }`}
        selected={selected}
      >
        <p className="pill-button-text secondary-font">{children}</p>
      </Pill>
    </button>
  );
}

export default PillButton;
