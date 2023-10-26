import Loading from '../Loading/Loading';
import useRequestInference from '../../hooks/useRequestInference';
import Score from '../Score/Score';
import './tellResult.css';
import PillButton from '../Pill/PillButton';

function TellResult({ file, onBack }) {
  const topValues = 5;
  const [isLoading, error, result, image] = useRequestInference(file);
  if (isLoading) {
    return <Loading />;
  }
  if (error) {
    return 'error';
  }
  return (
    <>
      <h3 className="main-font tell-result-heading">Your interior</h3>
      <img className="tell-result-image" src={image} alt="your-flat" />
      <h3 className="main-font tell-result-heading">design style is...</h3>
      <div className="tell-result-scores-container">
        {Object.entries(result).map((entry, idx) => {
          if (idx >= topValues) {
            return null;
          }
          const [styleName, styleScore] = entry;
          return (
            <Score
              fraction={styleScore}
              label={styleName}
              isMarked={idx === 0}
              key={styleName}
            />
          );
        })}
      </div>
      <PillButton className="tell-result-back-button" selected onClick={onBack}>
        Back
      </PillButton>
    </>
  );
}

export default TellResult;
