# External-tree package integration. The engine package is declared through
# Config.in / package/markos-engine; nothing else to hook here.
include $(sort $(wildcard $(BR2_EXTERNAL_MARKOS_PATH)/package/*/*.mk))
