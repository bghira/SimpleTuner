/**
 * Polyfills for JavaScript compatibility
 * Ensures modern JS features work in older browsers
 */

// Optional chaining polyfill helper
window.safeGet = function(obj, path, defaultValue = undefined) {
    const keys = path.split('.');
    let current = obj;

    for (const key of keys) {
        if (current == null) {
            return defaultValue;
        }
        current = current[key];
    }

    return current !== undefined ? current : defaultValue;
};

// Element.closest() polyfill for IE
if (!Element.prototype.closest) {
    Element.prototype.closest = function(selector) {
        var el = this;
        while (el && el.nodeType === 1) {
            if (el.matches(selector)) {
                return el;
            }
            el = el.parentElement || el.parentNode;
        }
        return null;
    };
}

// Element.matches() polyfill
if (!Element.prototype.matches) {
    Element.prototype.matches =
        Element.prototype.matchesSelector ||
        Element.prototype.mozMatchesSelector ||
        Element.prototype.msMatchesSelector ||
        Element.prototype.oMatchesSelector ||
        Element.prototype.webkitMatchesSelector ||
        function(s) {
            var matches = (this.document || this.ownerDocument).querySelectorAll(s),
                i = matches.length;
            while (--i >= 0 && matches.item(i) !== this) {}
            return i > -1;
        };
}

// Object.assign polyfill
if (typeof Object.assign !== 'function') {
    Object.defineProperty(Object, "assign", {
        value: function assign(target, varArgs) {
            'use strict';
            if (target === null || target === undefined) {
                throw new TypeError('Cannot convert undefined or null to object');
            }

            var to = Object(target);

            for (var index = 1; index < arguments.length; index++) {
                var nextSource = arguments[index];

                if (nextSource !== null && nextSource !== undefined) {
                    for (var nextKey in nextSource) {
                        if (Object.prototype.hasOwnProperty.call(nextSource, nextKey)) {
                            to[nextKey] = nextSource[nextKey];
                        }
                    }
                }
            }
            return to;
        },
        writable: true,
        configurable: true
    });
}

// Promise.finally polyfill
if (!Promise.prototype.finally) {
    Promise.prototype.finally = function(callback) {
        const constructor = this.constructor;
        return this.then(
            value => constructor.resolve(callback()).then(() => value),
            reason => constructor.resolve(callback()).then(() => { throw reason; })
        );
    };
}

// Array.includes polyfill
if (!Array.prototype.includes) {
    Array.prototype.includes = function(searchElement, fromIndex) {
        if (this == null) {
            throw new TypeError('"this" is null or not defined');
        }

        var O = Object(this);
        var len = parseInt(O.length) || 0;

        if (len === 0) {
            return false;
        }

        var n = parseInt(fromIndex) || 0;
        var k;

        if (n >= 0) {
            k = n;
        } else {
            k = len + n;
            if (k < 0) {
                k = 0;
            }
        }

        var currentElement;
        while (k < len) {
            currentElement = O[k];
            if (searchElement === currentElement ||
                (searchElement !== searchElement && currentElement !== currentElement)) {
                return true;
            }
            k++;
        }

        return false;
    };
}

// String.includes polyfill
if (!String.prototype.includes) {
    String.prototype.includes = function(search, start) {
        'use strict';

        if (search instanceof RegExp) {
            throw TypeError('first argument must not be a RegExp');
        }
        if (start === undefined) { start = 0; }
        return this.indexOf(search, start) !== -1;
    };
}

// Array.from polyfill
if (!Array.from) {
    Array.from = function(arrayLike, mapFn, thisArg) {
        var C = this;
        var items = Object(arrayLike);

        if (arrayLike == null) {
            throw new TypeError('Array.from requires an array-like object - not null or undefined');
        }

        var len = parseInt(items.length) || 0;
        var A = new Array(len);
        var k = 0;

        while (k < len) {
            if (k in items) {
                if (mapFn) {
                    A[k] = thisArg ? mapFn.call(thisArg, items[k], k) : mapFn(items[k], k);
                } else {
                    A[k] = items[k];
                }
            }
            k++;
        }

        A.length = len;
        return A;
    };
}

// NodeList.forEach polyfill
if (window.NodeList && !NodeList.prototype.forEach) {
    NodeList.prototype.forEach = function(callback, thisArg) {
        thisArg = thisArg || window;
        for (var i = 0; i < this.length; i++) {
            callback.call(thisArg, this[i], i, this);
        }
    };
}

// CustomEvent polyfill for IE
(function () {
    if (typeof window.CustomEvent === "function") return false;

    function CustomEvent(event, params) {
        params = params || { bubbles: false, cancelable: false, detail: null };
        var evt = document.createEvent('CustomEvent');
        evt.initCustomEvent(event, params.bubbles, params.cancelable, params.detail);
        return evt;
    }

    window.CustomEvent = CustomEvent;
})();

// FormData polyfill enhancements
if (window.FormData && !FormData.prototype.entries) {
    FormData.prototype.entries = function() {
        throw new Error('FormData.entries() is not supported in this browser. Please use a different approach.');
    };
}

// console.log safety for old IE
if (!window.console) {
    window.console = {};
}
if (!window.console.log) {
    window.console.log = function() {};
}
if (!window.console.error) {
    window.console.error = function() {};
}
if (!window.console.warn) {
    window.console.warn = function() {};
}
if (!window.console.info) {
    window.console.info = function() {};
}

// ES6 module loading fallback
window.moduleLoader = {
    modules: {},

    define: function(name, dependencies, factory) {
        if (typeof dependencies === 'function') {
            factory = dependencies;
            dependencies = [];
        }

        var module = { exports: {} };
        var deps = dependencies.map(function(dep) {
            if (dep === 'exports') return module.exports;
            return window[dep] || window.moduleLoader.modules[dep];
        });

        factory.apply(null, deps);
        window.moduleLoader.modules[name] = module.exports;
        return module.exports;
    },

    require: function(name) {
        return window.moduleLoader.modules[name];
    }
};

// Export compatibility helpers
window.compat = {
    safeGet: window.safeGet,
    moduleLoader: window.moduleLoader
};