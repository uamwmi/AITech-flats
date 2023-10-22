import getProtectedEndpointUrl from './url';

export const postRequest = (url, data) => {
  const params = {
    method: 'POST',
    body: JSON.stringify(data),
    headers: {
      'Content-Type': 'application/json',
    },
  };
  return fetch(getProtectedEndpointUrl(url), params);
};

export const getLocalRequest = (path) => {
  const params = {
    method: 'GET',
  };
  return fetch(`./${path}`, params);
};

export default { postRequest, getLocalRequest };
