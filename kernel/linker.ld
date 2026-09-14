/*
 * MarkOS kernel linker script.
 *
 * The kernel is higher-half linked at the start of the topmost 2 GiB of the
 * virtual address space, as the Limine boot protocol mandates. Limine reads
 * the program headers to decide MMU permissions per segment, so the Limine
 * requests, .text, .rodata and .data each get their own PT_LOAD.
 *
 * Mirrors the official limine-c-template x86_64 script (pinned Limine v12.9.0).
 */

OUTPUT_FORMAT(elf64-x86-64)

ENTRY(kernel_main)

PHDRS
{
    limine_requests PT_LOAD;
    text            PT_LOAD;
    rodata          PT_LOAD;
    data            PT_LOAD;
}

SECTIONS
{
    . = 0xffffffff80000000;

    /* Limine scans the loaded image for the request start/end marker magic
     * values; keeping these three input sections contiguous and in this
     * order guarantees every request lies between the markers. */
    .limine_requests : {
        KEEP(*(.limine_requests_start))
        KEEP(*(.limine_requests))
        KEEP(*(.limine_requests_end))
    } :limine_requests

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .text : {
        *(.text .text.*)
    } :text

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .rodata : {
        *(.rodata .rodata.*)
    } :rodata

    .note.gnu.build-id : {
        *(.note.gnu.build-id)
    } :rodata

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .data : {
        *(.data .data.*)
    } :data

    /* .bss must be the last :data section so no zero padding lands in the
     * image file; the bootloader overwrites it with zeroes at load time. */
    .bss : {
        *(.bss .bss.*)
        *(COMMON)
    } :data

    /DISCARD/ : {
        *(.eh_frame*)
        *(.note .note.*)
    }
}
