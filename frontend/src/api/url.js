const API_URL = import.meta.env.VITE_API_URL;
const API_TOKEN = import.meta.env.VITE_API_TOKEN;

const getProtectedEndpointUrl = (url) => {
  return `${API_URL}${url}?code=${API_TOKEN}`;
};

export default getProtectedEndpointUrl;
