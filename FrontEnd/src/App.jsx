import React, { useState } from 'react';
import FormPage from './FormPage';
import HistopathologyDashboard from './Histopathology';

function App() {
  const [result, setResult] = useState(null);

  return (
    <>
      {result ? (
        <HistopathologyDashboard data={result} />
      ) : (
        <FormPage onResult={setResult} />
      )}
    </>
  );
}

export default App;
