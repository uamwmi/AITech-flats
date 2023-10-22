module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  extends: ['airbnb', 'airbnb/hooks', 'plugin:prettier/recommended'],
  overrides: [],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  plugins: ['react', 'prettier'],
  rules: {
    'react/react-in-jsx-scope': 0,
    'prettier/prettier': 'error',
    'react/prop-types': 0,
    'react/button-has-type': 0,
    'import/no-extraneous-dependencies': ['error', {'devDependencies': true}],
    'jsx-a11y/no-noninteractive-element-interactions': 0,
    'jsx-a11y/no-noninteractive-tabindex': 0
  },
};
