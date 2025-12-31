/**
 * Jest configuration for Alpine.js component tests.
 *
 * Run tests with: npx jest --config tests/js/jest.config.js
 */

module.exports = {
    testEnvironment: 'jsdom',
    rootDir: '../../',
    testMatch: ['<rootDir>/tests/js/**/*.test.js'],
    setupFilesAfterEnv: ['<rootDir>/tests/js/setup.js'],
    moduleNameMapper: {
        '^@/(.*)$': '<rootDir>/simpletuner/static/js/$1',
    },
    modulePathIgnorePatterns: [
        '<rootDir>/.venv/',
        '<rootDir>/node_modules/',
    ],
    collectCoverageFrom: [
        'simpletuner/static/js/**/*.js',
        '!simpletuner/static/js/vendor/**',
    ],
    coverageDirectory: 'coverage/js',
    coverageReporters: ['text', 'lcov'],
    verbose: true,
};
