/*
 * MarkOS AArch64 linker script.
 *
 * Flat physical layout at 0x80000 — the Pi firmware's 64-bit kernel load
 * address (QEMU raspi machines load there too). No MMU yet: physical ==
 * virtual during bring-up; the paging phase keeps the identity map for
 * peripheral space and adds proper kernel mappings.
 */

ENTRY(_start)

SECTIONS
{
    . = 0x80000;
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
