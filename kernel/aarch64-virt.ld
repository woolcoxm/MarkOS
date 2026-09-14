/*
 * MarkOS AArch64 linker script — QEMU `virt` board variant.
 *
 * Same layout as aarch64.ld but linked at QEMU virt's 64-bit kernel load
 * address (DRAM base 0x40000000 + text offset 0x80000).
 */

ENTRY(_start)

SECTIONS
{
    . = 0x40080000;
    __image_start = .;

    .text : {
        KEEP(*(.text.boot))
        *(.text .text.*)
    }

    .rodata : ALIGN(8) {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(8) {
        *(.data .data.* .got .got.*)
    }

    .bss : ALIGN(16) {
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.*)
        *(COMMON)
        . = ALIGN(16);
        __bss_end = .;
    }

    /* Boot stack, grown by symbol arithmetic only — nothing is emitted. */
    . = ALIGN(16);
    __stack_bottom = .;
    . += 0x10000;                /* 64 KiB */
    __stack_top = .;

    __image_end = .;

    /DISCARD/ : {
        *(.eh_frame*)
        *(.note .note.*)
    }
}
