module.exports = {
  webpack: {
    configure: {
      resolve: {
        fallback: {
          "os": require.resolve("os-browserify/browser"),
          "util": require.resolve("util/"),
          "path": require.resolve("path-browserify"),
          "fs": false,
          "stream": require.resolve("stream-browserify"),
          "buffer": require.resolve("buffer/"),
          "crypto": require.resolve("crypto-browserify"),
          "assert": require.resolve("assert/")
        }
      }
    }
  }
}; 