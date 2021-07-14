# usm
### Micro State Machine for C++11 ###

This is a lightweight state machine, or more some tools to help you make your own classes into well organized state machines. The state transition design is specified in a declarative manner, allowing humans to have a quick overview of all of the states, transitions and fallbacks configured.

### Performance ###
It is configured entirely at compile time, requires no dynamic memory allocations or vtable lookups, and should be extremely fast and lightweight.

### Helpers ###
There is also the capability to generate `.dot` files for visualizing the state flow transitions, by parsing the configuration table out of C++ code. Check out what it looks like [here](https://bit.ly/2LyIrXg)!

### Sounds good, how do I use it? ###
There is a full example in the tests. Otherwise you can check out [PX4/Avoidance](https://github.com/PX4/avoidance) where it is used in several places.
