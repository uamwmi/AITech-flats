import React, { Suspense, useState } from 'react';
import AppLayout from '../layouts/AppLayout/AppLayout';
import Header from '../components/Header/Header';
import MainForm from '../components/MainForm/MainForm';
import Loading from '../components/Loading/Loading';

const TellResult = React.lazy(() =>
  import('../components/TellResult/TellResult')
);

function TellPage() {
  const defaultFileState = { name: '', data: null };
  const [file, setFile] = useState({ ...defaultFileState });
  const resetFile = () => setFile({ ...defaultFileState });
  const onSubmit = (f) => {
    setFile(f);
  };
  return (
    <AppLayout>
      {!file.data ? (
        <>
          <Header
            heading="INTER.IO/TELL"
            subheading="What architectural style is this interior?"
          />
          <MainForm onSubmit={onSubmit} />
        </>
      ) : (
        <Suspense fallback={<Loading />}>
          <TellResult file={file} onBack={resetFile} />
        </Suspense>
      )}
    </AppLayout>
  );
}

export default TellPage;
