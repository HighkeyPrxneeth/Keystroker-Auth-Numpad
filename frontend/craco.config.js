module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Add any custom webpack configuration here
      return webpackConfig;
    },
  },
  devServer: {
    open: false,
    port: 3000,
  },
};
