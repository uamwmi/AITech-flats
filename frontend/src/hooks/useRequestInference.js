import { useEffect, useRef, useState } from 'react';
import { postRequest } from '../api/request';

const useRequestInference = (file) => {
  const [isLoading, setIsLoading] = useState(true);
  const [result, setResult] = useState({});
  const [error, setError] = useState(false);
  const [image, setImage] = useState(null);
  const once = useRef(false);
  useEffect(() => {
    if (once.current) {
      return () => {};
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const response = await postRequest('score', {
          image: { data: reader.result.split(',')[1] },
        });
        const data = await response.json();
        setIsLoading(false);
        setResult(data);
        setImage(reader.result);
      } catch (e) {
        setIsLoading(false);
        setError(true);
      }
    };
    reader.readAsDataURL(file.data);
    return () => {
      once.current = true;
    };
  }, [file]);
  return [isLoading, error, result, image];
};

export default useRequestInference;
