//! Board description layer: per-board constants, selected by cargo feature.
//!
//! Two boards are supported:
//!  - `board-pi`  (default): Raspberry Pi (QEMU raspi3b dev machine and real
//!    Pi 4/5 hardware). Legacy peripheral window, spin-table/mailbox release.
//!  - `board-virt`: QEMU `virt` machine — architecturally standard board
//!    (PL011 at 0x09000000, GIC-400, PSCI CPU_ON release). Used for the
//!    automated SMP/GIC acceptance tests.
//!
//! invariants: exactly one board feature must be enabled.

#[cfg(all(feature = "board-virt", feature = "board-pi"))]
compile_error!("select exactly one board feature");

#[cfg(feature = "board-virt")]
pub const UART_BASE: usize = 0x0900_0000;
#[cfg(feature = "board-virt")]
pub const CORE_COUNT: usize = 4;
#[cfg(feature = "board-virt")]
pub const NAME: &str = "qemu-virt";

#[cfg(feature = "board-pi")]
pub const UART_BASE: usize = 0x3F20_1000;
#[cfg(feature = "board-pi")]
pub const CORE_COUNT: usize = 4;
#[cfg(feature = "board-pi")]
pub const NAME: &str = "raspi (pi3-qemu / pi4-pi5 hw)";
