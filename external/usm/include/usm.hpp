#pragma once

/*
* Copyright (c) 2019 Julian Kent. All rights reserved.
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of libgnc nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*
*
* For usage examples, look at the tests.
*/

#include <cstdint>

namespace usm {

enum class DefaultTransition { REPEAT, NEXT, ERROR };

template <typename StateEnum, typename TransitionEnum = DefaultTransition>
class StateMachine {
 public:
  using State = StateEnum;
  using Transition = TransitionEnum;

  StateMachine(State startingState);
  void iterateOnce();

  State getState();
  uint64_t stateRunCount();

 protected:
  virtual Transition runCurrentState(State currentState) = 0;  // a big switch
  virtual State chooseNextState(State currentState, Transition transition) = 0;  // nested switches

 private:
  StateEnum m_currentState;
  uint64_t m_stateRunCount;
};

/*---------------IMPLEMENTATION------------------*/

template <typename StateEnum, typename TransitionEnum>
StateMachine<StateEnum, TransitionEnum>::StateMachine(State startingState)
    : m_currentState(startingState), m_stateRunCount(0) {}

template <typename StateEnum, typename TransitionEnum>
void StateMachine<StateEnum, TransitionEnum>::iterateOnce() {
  Transition t = runCurrentState(m_currentState);
  m_stateRunCount++;
  StateEnum prev_state = m_currentState;
  if (t != Transition::REPEAT) m_currentState = chooseNextState(m_currentState, t);
  if (m_currentState != prev_state) m_stateRunCount = 0;
}

template <typename StateEnum, typename TransitionEnum>
StateEnum StateMachine<StateEnum, TransitionEnum>::getState() {
  return m_currentState;
}

template <typename StateEnum, typename TransitionEnum>
uint64_t StateMachine<StateEnum, TransitionEnum>::stateRunCount() {
    return m_stateRunCount;
}

}

/*---------------MACROS TO MAKE TRANSITION TABLES EASY------------------*/

// clang-format off
#define USM_TABLE(current_state, error, ...) \
switch (current_state) { \
    __VA_ARGS__; \
    default: break; \
} \
return error

#define USM_STATE(transition, start_state, ...) \
    case start_state: \
        switch (transition) { \
            __VA_ARGS__; \
            default: break; \
        } \
    break

#define USM_MAP(transition, next_state) \
            case transition: return next_state

// clang-format on
