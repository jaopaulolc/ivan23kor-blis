#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of The University of Texas at Austin nor the names
#     of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# Makefile
#
# Field G. Van Zee
# 
# Top-level makefile for libflame linear algebra library.
#
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all libs test install uninstall uninstall-old clean \
        check-env check-env-mk check-env-fragments check-env-make-defs \
        testsuite testsuite-bin testsuite-run \
        flat-header flat-cblas-header \
        install-libs install-headers install-lib-symlinks \
        showconfig \
        cleanlib cleanh cleantest cleanmk distclean \
        changelog \
        uninstall-libs uninstall-headers uninstall-lib-symlinks \
        uninstall-old-libs uninstall-old-headers



#
# --- Include common makefile definitions --------------------------------------
#

# Define the name of the common makefile.
COMMON_MK_FILE    := common.mk

# Construct the path to the makefile configuration file that was generated by
# the configure script.
COMMON_MK_PATH    := $(COMMON_MK_FILE)

# Include the configuration file.
-include $(COMMON_MK_FILE)

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(COMMON_MK_INCLUDED)),yes)
COMMON_MK_PRESENT := yes
else
COMMON_MK_PRESENT := no
endif



#
# --- Main target variable definitions -----------------------------------------
#

# --- Object/library paths ---

# Construct the base object file path for the current configuration.
BASE_OBJ_PATH          := ./$(OBJ_DIR)/$(CONFIG_NAME)

# Construct base object file paths corresponding to the four locations
# of source code.
BASE_OBJ_CONFIG_PATH   := $(BASE_OBJ_PATH)/$(CONFIG_DIR)
BASE_OBJ_FRAME_PATH    := $(BASE_OBJ_PATH)/$(FRAME_DIR)
BASE_OBJ_REFKERN_PATH  := $(BASE_OBJ_PATH)/$(REFKERN_DIR)
BASE_OBJ_KERNELS_PATH  := $(BASE_OBJ_PATH)/$(KERNELS_DIR)

# Construct the base path for the library.
BASE_LIB_PATH          := ./$(LIB_DIR)/$(CONFIG_NAME)

# Construct the architecture-version string, which will be used to name the
# library upon installation.
VERS_CONF              := $(VERSION)-$(CONFIG_NAME)

# --- Library names ---

# Note: These names will be modified later to include the configuration and
# version strings.
BLIS_LIB_NAME          := $(BLIS_LIB_BASE_NAME).a
BLIS_DLL_NAME          := $(BLIS_LIB_BASE_NAME).so

# Append the base library path to the library names.
BLIS_LIB_PATH          := $(BASE_LIB_PATH)/$(BLIS_LIB_NAME)
BLIS_DLL_PATH          := $(BASE_LIB_PATH)/$(BLIS_DLL_NAME)

# --- BLIS framework object variable names ---

# These hold object filenames corresponding to above.
MK_FRAME_OBJS          :=
MK_REFKERN_OBJS        :=
MK_KERNELS_OBJS        :=

# --- Define install target names for static libraries ---

MK_BLIS_LIB                  := $(BLIS_LIB_PATH)
MK_BLIS_LIB_INST             := $(patsubst $(BASE_LIB_PATH)/%.a, \
                                           $(INSTALL_PREFIX)/lib/%.a, \
                                           $(MK_BLIS_LIB))
MK_BLIS_LIB_INST_W_VERS_CONF := $(patsubst $(BASE_LIB_PATH)/%.a, \
                                           $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a, \
                                           $(MK_BLIS_LIB))

# --- Define install target names for shared libraries ---

MK_BLIS_DLL                  := $(BLIS_DLL_PATH)
MK_BLIS_DLL_INST             := $(patsubst $(BASE_LIB_PATH)/%.so, \
                                           $(INSTALL_PREFIX)/lib/%.so, \
                                           $(MK_BLIS_DLL))
MK_BLIS_DLL_INST_W_VERS_CONF := $(patsubst $(BASE_LIB_PATH)/%.so, \
                                           $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).so, \
                                           $(MK_BLIS_DLL))

# --- Determine which libraries to build ---

MK_LIBS                           :=
MK_LIBS_INST                      :=
MK_LIBS_INST_W_VERS_CONF          :=

ifeq ($(BLIS_ENABLE_STATIC_BUILD),yes)
MK_LIBS                           += $(MK_BLIS_LIB)
MK_LIBS_INST                      += $(MK_BLIS_LIB_INST)
MK_LIBS_INST_W_VERS_CONF          += $(MK_BLIS_LIB_INST_W_VERS_CONF)
endif

ifeq ($(BLIS_ENABLE_DYNAMIC_BUILD),yes)
MK_LIBS                           += $(MK_BLIS_DLL)
MK_LIBS_INST                      += $(MK_BLIS_DLL_INST)
MK_LIBS_INST_W_VERS_CONF          += $(MK_BLIS_DLL_INST_W_VERS_CONF)
endif

# Strip leading, internal, and trailing whitespace.
MK_LIBS_INST                      := $(strip $(MK_LIBS_INST))
MK_LIBS_INST_W_VERS_CONF          := $(strip $(MK_LIBS_INST_W_VERS_CONF))

# Set the include directory names
MK_INCL_DIR_INST                  := $(INSTALL_PREFIX)/include/blis



#
# --- Library object definitions -----------------------------------------------
#

# In this section, we will isolate the relevant source code filepaths and
# convert them to lists of object filepaths. Relevant source code falls into
# four categories: configuration source; architecture-specific kernel source;
# reference kernel source; and general framework source.

# $(call gen-obj-paths-from-src file_exts, src_files, base_src_path, base_obj_path)
#gen-obj-paths-from-src = $(foreach ch, $(1), \
#                             $(patsubst $(3)/%.$(ch), \
#                                        $(4)/%.o, \
#                                        $(2) \
#                              ) \
#                          )

# First, identify the source code found in the configuration sub-directories.
MK_CONFIG_C          := $(filter %.c, $(MK_CONFIG_SRC))
MK_CONFIG_S          := $(filter %.s, $(MK_CONFIG_SRC))
MK_CONFIG_SS         := $(filter %.S, $(MK_CONFIG_SRC))
MK_CONFIG_C_OBJS     := $(patsubst $(CONFIG_PATH)/%.c, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                   $(MK_CONFIG_C) \
                         )
MK_CONFIG_S_OBJS     := $(patsubst $(CONFIG_PATH)/%.s, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                   $(MK_CONFIG_S) \
                         )
MK_CONFIG_SS_OBJS    := $(patsubst $(CONFIG_PATH)/%.S, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                   $(MK_CONFIG_SS) \
                         )
MK_CONFIG_OBJS       := $(MK_CONFIG_C_OBJS) \
                        $(MK_CONFIG_S_OBJS) \
                        $(MK_CONFIG_SS_OBJS)

# A more concise but obfuscated way of encoding the above lines.
#MK_CONFIG_OBJS       := $(call gen-obj-paths-from-src c s S,
#                                                      $(MK_CONFIG_SRC),
#                                                      $(CONFIG_PATH),
#                                                      $(BASE_OBJ_CONFIG_PATH)
#                         )

# Now, identify all of the architecture-specific kernel source code. We
# start by filtering only .c and .[sS] files (ignoring any .h files, though
# there shouldn't be any), and then instantiating object file paths from the
# source file paths. Note that MK_KERNELS_SRC is already limited to the
# kernel source corresponding to the kernel sets in KERNEL_LIST. This
# is because the configure script only propogated makefile fragments into
# those specific kernel subdirectories.
MK_KERNELS_C       := $(filter %.c, $(MK_KERNELS_SRC))
MK_KERNELS_S       := $(filter %.s, $(MK_KERNELS_SRC))
MK_KERNELS_SS      := $(filter %.S, $(MK_KERNELS_SRC))
MK_KERNELS_C_OBJS  := $(patsubst $(KERNELS_PATH)/%.c, $(BASE_OBJ_KERNELS_PATH)/%.o, \
                                 $(MK_KERNELS_C) \
                       )
MK_KERNELS_S_OBJS  := $(patsubst $(KERNELS_PATH)/%.s, $(BASE_OBJ_KERNELS_PATH)/%.o, \
                                 $(MK_KERNELS_S) \
                       )
MK_KERNELS_SS_OBJS := $(patsubst $(KERNELS_PATH)/%.S, $(BASE_OBJ_KERNELS_PATH)/%.o, \
                                 $(MK_KERNELS_SS) \
                       )
MK_KERNELS_OBJS    := $(MK_KERNELS_C_OBJS) \
                      $(MK_KERNELS_S_OBJS) \
                      $(MK_KERNELS_SS_OBJS)

# Next, identify all of the reference kernel source code, then filter only
# .c files (ignoring .h files), and finally instantiate object file paths
# from the source files paths once for each sub-configuration in CONFIG_LIST,
# appending the name of the sub-config to the object filename.
MK_REFKERN_C       := $(filter %.c, $(MK_REFKERN_SRC))
MK_REFKERN_OBJS    := $(foreach arch, $(CONFIG_LIST), \
                          $(patsubst $(REFKERN_PATH)/%_$(REF_SUF).c, \
                                     $(BASE_OBJ_REFKERN_PATH)/$(arch)/%_$(arch)_$(REF_SUF).o, \
                                     $(MK_REFKERN_C) \
                           ) \
                       )

# And now, identify all of the portable framework source code, then filter
# only .c files (ignoring .h files), and finally instantiate object file
# paths from the source file paths.
MK_FRAME_C         := $(filter %.c, $(MK_FRAME_SRC))
MK_FRAME_OBJS      := $(patsubst $(FRAME_PATH)/%.c, $(BASE_OBJ_FRAME_PATH)/%.o, \
                                 $(MK_FRAME_C) \
                       )

# Combine all of the object files into some readily-accessible variables.
MK_BLIS_OBJS         := $(MK_CONFIG_OBJS) \
                        $(MK_KERNELS_OBJS) \
                        $(MK_REFKERN_OBJS) \
                        $(MK_FRAME_OBJS)

# Optionally filter out the BLAS and CBLAS compatibility layer object files.
# This is not actually necessary, since each affected file is guarded by C
# preprocessor macros, but it but prevents "empty" object files from being
# added into the library (and reduces compilation time).
BASE_OBJ_BLAS_PATH   := $(BASE_OBJ_FRAME_PATH)/compat
BASE_OBJ_CBLAS_PATH  := $(BASE_OBJ_FRAME_PATH)/compat/cblas
ifeq ($(BLIS_ENABLE_CBLAS),no)
MK_BLIS_OBJS         := $(filter-out $(BASE_OBJ_CBLAS_PATH)/%.o, $(MK_BLIS_OBJS) )
endif
ifeq ($(BLIS_ENABLE_BLAS2BLIS),no)
MK_BLIS_OBJS         := $(filter-out $(BASE_OBJ_BLAS_PATH)/%.o,  $(MK_BLIS_OBJS) )
endif



#
# --- Monolithic header definitions --------------------------------------------
#

# Define a list of headers to install. The default is to only install blis.h.
HEADERS_TO_INSTALL := $(BLIS_H_FLAT)

# If CBLAS is enabled, we also install cblas.h so the user does not need to
# change their source code to #include "blis.h" in order to access the CBLAS
# function prototypes and enums.
ifeq ($(BLIS_ENABLE_CBLAS),yes)
HEADERS_TO_INSTALL += $(CBLAS_H_FLAT)
endif



#
# --- Test suite definitions ---------------------------------------------------
#

# The location of the test suite's general and operations-specific
# input/configuration files.
TESTSUITE_CONF_GEN_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_CONF_GEN)
TESTSUITE_CONF_OPS_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_CONF_OPS)

# The locations of the test suite source directory and the local object
# directory.
TESTSUITE_SRC_PATH      := $(DIST_PATH)/$(TESTSUITE_DIR)/src
BASE_OBJ_TESTSUITE_PATH := $(BASE_OBJ_PATH)/$(TESTSUITE_DIR)

# Convert source file paths to object file paths by replacing the base source
# directories with the base object directories, and also replacing the source
# file suffix (eg: '.c') with '.o'.
MK_TESTSUITE_OBJS       := $(strip \
                           $(patsubst $(TESTSUITE_SRC_PATH)/%.c, \
                                      $(BASE_OBJ_TESTSUITE_PATH)/%.o, \
                                      $(wildcard $(TESTSUITE_SRC_PATH)/*.c)) \
                            )

# The test suite binary executable filename.
TESTSUITE_BIN           := test_libblis.x



#
# --- Uninstall definitions ----------------------------------------------------
#

# This shell command grabs all files named "libblis-*.a" or "libblis-*.so" in
# the installation directory and then filters out the name of the library
# archive for the current version/configuration and its symlink. We consider
# this remaining set of libraries to be "old" and eligible for removal upon
# running of the uninstall-old target.
UNINSTALL_LIBS    := $(shell $(FIND) $(INSTALL_PREFIX)/lib/ -name "$(BLIS_LIB_BASE_NAME)-*.a" 2> /dev/null | $(GREP) -v "$(BLIS_LIB_BASE_NAME)-$(VERS_CONF).a" | $(GREP) -v $(BLIS_LIB_NAME))
UNINSTALL_LIBS    += $(shell $(FIND) $(INSTALL_PREFIX)/lib/ -name "$(BLIS_LIB_BASE_NAME)-*.so" 2> /dev/null | $(GREP) -v "$(BLIS_LIB_BASE_NAME)-$(VERS_CONF).so" | $(GREP) -v $(BLIS_LIB_NAME))

# This shell command grabs all files named "*.h" that are not blis.h or cblas.h
# in the installation directory. We consider this set of headers to be "old" and
# eligible for removal upon running of the uninstall-old-headers target.
UNINSTALL_HEADERS := $(shell $(FIND) $(INSTALL_PREFIX)/include/blis/ -name "*.h" 2> /dev/null | $(GREP) -v "$(BLIS_H)" | $(GREP) -v "$(CBLAS_H)")



#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: libs

libs: blis-lib

test: testsuite

install: libs install-libs install-headers install-lib-symlinks

uninstall: uninstall-libs uninstall-headers uninstall-lib-symlinks

uninstall-old: uninstall-old-libs uninstall-old-headers

clean: cleanlib cleantest


# --- Consolidated blis.h header creation ---

flat-header: check-env $(BLIS_H_FLAT)

$(BLIS_H_FLAT): $(MK_HEADER_FILES)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(FLATTEN_H) -c -v1 $(BLIS_H_SRC_PATH) $@ "./$(INCLUDE_DIR)" "$(MK_HEADER_DIR_PATHS)"
else
	@echo -n "Generating monolithic blis.h"
	@$(FLATTEN_H) -c -v1 $(BLIS_H_SRC_PATH) $@ "./$(INCLUDE_DIR)" "$(MK_HEADER_DIR_PATHS)"
	@echo "Generated $@"
endif

# --- Consolidated cblas.h header creation ---

flat-cblas-header: check-env $(CBLAS_H_FLAT)

$(CBLAS_H_FLAT): $(MK_HEADER_FILES)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(FLATTEN_H) -c -v1 $(CBLAS_H_SRC_PATH) $@ "./$(INCLUDE_DIR)" "$(MK_HEADER_DIR_PATHS)"
else
	@echo -n "Generating monolithic cblas.h"
	@$(FLATTEN_H) -c -v1 $(CBLAS_H_SRC_PATH) $@ "./$(INCLUDE_DIR)" "$(MK_HEADER_DIR_PATHS)"
	@echo "Generated $@"
endif


# --- General source code / object code rules ---

# FGVZ: Add support for compiling .s and .S files in 'config'/'kernels'
# directories.
#  - May want to add an extra foreach loop around function eval/call.

# first argument: a configuration name from config_list, used to look up the
# CFLAGS to use during compilation.
define make-config-rule
$(BASE_OBJ_CONFIG_PATH)/$(1)/%.o: $(CONFIG_PATH)/$(1)/%.c $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(call get-config-cflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-config-text-for,$(1))
	@$(CC) $(call get-config-cflags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a configuration name from the union of config_list and
# config_name, used to look up the CFLAGS to use during compilation.
define make-frame-rule
$(BASE_OBJ_FRAME_PATH)/%.o: $(FRAME_PATH)/%.c $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(call get-frame-cflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-frame-text-for,$(1))
	@$(CC) $(call get-frame-cflags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a kernel set (name) being targeted (e.g. haswell).
define make-refkern-rule
$(BASE_OBJ_REFKERN_PATH)/$(1)/%_$(1)_ref.o: $(REFKERN_PATH)/%_ref.c $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(call get-refkern-cflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refkern-text-for,$(1))
	@$(CC) $(call get-refkern-cflags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a kernel set (name) being targeted (e.g. haswell).
# second argument: the configuration whose CFLAGS we should use in compilation.
# third argument: the kernel file suffix being considered.
#$(BASE_OBJ_KERNELS_PATH)/$(1)/%.o: $(KERNELS_PATH)/$(1)/%.$(3) $(MK_HEADER_FILES) $(MAKE_DEFS_MK_PATHS)
define make-kernels-rule
$(BASE_OBJ_KERNELS_PATH)/$(1)/%.o: $(KERNELS_PATH)/$(1)/%.c $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(call get-kernel-cflags-for,$(2)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-kernel-text-for,$(1))
	@$(CC) $(call get-kernel-cflags-for,$(2)) -c $$< -o $$@
endif
endef

# Define functions to choose the correct sub-configuration name for the
# given kernel set. This function is called when instantiating the
# make-kernels-rule.
get-config-for-kset = $(lastword $(subst :, ,$(filter $(1):%,$(KCONFIG_MAP))))

# Instantiate the build rule for files in the configuration directory for
# each of the sub-configurations in CONFIG_LIST with the CFLAGS designated
# for that sub-configuration.
$(foreach conf, $(CONFIG_LIST), $(eval $(call make-config-rule,$(conf))))

# Instantiate the build rule for non-kernel framework files. Use the CFLAGS for
# the configuration family, which exists in the directory whose name is equal to
# CONFIG_NAME. (BTW: If it is a singleton family, then CONFIG_NAME is equal to
# CONFIG_LIST.)
#$(eval $(call make-frame-rule,$(firstword $(CONFIG_NAME))))
$(foreach conf, $(CONFIG_NAME), $(eval $(call make-frame-rule,$(conf))))

# Instantiate the build rule for reference kernels for each of the sub-
# configurations in CONFIG_LIST with the CFLAGS designated for that sub-
# configuration.
$(foreach conf, $(CONFIG_LIST), $(eval $(call make-refkern-rule,$(conf))))

# Instantiate the build rule for optimized kernels for each of the kernel
# sets in KERNEL_LIST with the CFLAGS designated for the sub-configuration
# specified by the KCONFIG_MAP.
$(foreach kset, $(KERNEL_LIST), $(eval $(call make-kernels-rule,$(kset),$(call get-config-for-kset,$(kset)))))

# FGVZ: for later, to compile multiple kernel source suffixes.
#$(foreach suf,  $(KERNEL_SUFS), \
#$(foreach kset, $(KERNEL_LIST), $(eval $(call make-kernels-rule,$(kset),$(suf)))))


# --- Environment check rules ---

check-env: check-env-make-defs check-env-fragments check-env-mk

check-env-mk:
$(info CONFIG_MK_PRESENT is '$(CONFIG_MK_PRESENT)')
ifeq ($(CONFIG_MK_PRESENT),no)
	$(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-env-fragments: check-env-mk
ifeq ($(MAKEFILE_FRAGMENTS_PRESENT),no)
	$(error Cannot proceed: makefile fragments not detected! Run configure first)
endif

check-env-make-defs: check-env-fragments
ifeq ($(ALL_MAKE_DEFS_MK_PRESENT),no)
	$(error Cannot proceed: Some make_defs.mk files not found or mislabeled!)
endif


# --- All-purpose library rule (static and shared) ---

blis-lib: check-env $(MK_LIBS)


# --- Static library archiver rules ---

$(MK_BLIS_LIB): $(MK_BLIS_OBJS)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@
else
	@echo "Archiving $@"
	@$(AR) $(ARFLAGS) $@ $?
	@$(RANLIB) $@
endif


# --- Dynamic library linker rules ---

$(MK_BLIS_DLL): $(MK_BLIS_OBJS)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(LINKER) $(SOFLAGS) $(LDFLAGS) -o $@ $?
else 
	@echo "Dynamically linking $@"
	@$(LINKER) $(SOFLAGS) $(LDFLAGS) -o $@ $?
endif


# --- Test suite rules ---

testsuite: testsuite-run

testsuite-bin: check-env $(TESTSUITE_BIN)

$(BASE_OBJ_TESTSUITE_PATH)/%.o: $(TESTSUITE_SRC_PATH)/%.c
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(call get-frame-cflags-for,$(CONFIG_NAME)) -c $< -o $@
else
	@echo "Compiling $<"
	@$(CC) $(call get-frame-cflags-for,$(CONFIG_NAME)) -c $< -o $@
endif

$(TESTSUITE_BIN): $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(LINKER) $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(LDFLAGS) -o $@
else
	@echo "Linking $@ against '$(MK_BLIS_LIB) $(LDFLAGS)'"
	@$(LINKER) $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(LDFLAGS) -o $@
endif


testsuite-run: testsuite-bin
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	./$(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                   -o $(TESTSUITE_CONF_OPS_PATH) \
                        > $(TESTSUITE_OUT_FILE)

else
	@echo "Running $(TESTSUITE_BIN) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@./$(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                    -o $(TESTSUITE_CONF_OPS_PATH) \
                         > $(TESTSUITE_OUT_FILE)
endif


# --- Install header rules ---

install-headers: check-env $(MK_INCL_DIR_INST)

$(MK_INCL_DIR_INST): $(HEADERS_TO_INSTALL) $(CONFIG_MK_FILE)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(MKDIR) $(@)
	$(INSTALL) -m 0644 $(HEADERS_TO_INSTALL) $(@)
else
	@$(MKDIR) $(@)
	@echo "Installing $(notdir $(HEADERS_TO_INSTALL)) into $(@)/"
	@$(INSTALL) -m 0644 $(HEADERS_TO_INSTALL) $(@)
endif


# --- Install library rules ---

install-libs: check-env $(MK_LIBS_INST_W_VERS_CONF)

$(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a: $(BASE_LIB_PATH)/%.a $(CONFIG_MK_FILE)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(MKDIR) $(@D)
	$(INSTALL) -m 0644 $< $@
else
	@echo "Installing $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $< $@
endif

$(INSTALL_PREFIX)/lib/%-$(VERS_CONF).so: $(BASE_LIB_PATH)/%.so $(CONFIG_MK_FILE)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(MKDIR) $(@D)
	$(INSTALL) -m 0644 $< $@
else
	@echo "Installing $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $< $@
endif


# --- Install-symlinks rules ---

install-lib-symlinks: check-env $(MK_LIBS_INST)

$(INSTALL_PREFIX)/lib/%.a: $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(SYMLINK) $(<F) $(@F)
	$(MV) $(@F) $(INSTALL_PREFIX)/lib/
else
	@echo "Installing symlink $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(SYMLINK) $(<F) $(@F)
	@$(MV) $(@F) $(INSTALL_PREFIX)/lib/
endif

$(INSTALL_PREFIX)/lib/%.so: $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).so
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(SYMLINK) $(<F) $(@F)
	$(MV) $(@F) $(INSTALL_PREFIX)/lib/
else
	@echo "Installing symlink $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(SYMLINK) $(<F) $(@F)
	@$(MV) $(@F) $(INSTALL_PREFIX)/lib/
endif


# --- Query current configuration ---

showconfig: check-env
	@echo "configuration family:  $(CONFIG_NAME)"
	@echo "sub-configurations:    $(CONFIG_LIST)"
	@echo "requisite kernels:     $(KERNEL_LIST)"
	@echo "kernel-to-config map:  $(KCONFIG_MAP)"
	@echo "-----------------------"
	@echo "BLIS version string:   $(VERSION)"
	@echo "install prefix:        $(INSTALL_PREFIX)"
	@echo "debugging status:      $(DEBUG_TYPE)"
	@echo "multithreading status: $(THREADING_MODEL)"
	@echo "enable BLAS API?       $(BLIS_ENABLE_BLAS2BLIS)"
	@echo "enable CBLAS API?      $(BLIS_ENABLE_CBLAS)"
	@echo "build static library?  $(BLIS_ENABLE_STATIC_BUILD)"
	@echo "build shared library?  $(BLIS_ENABLE_DYNAMIC_BUILD)"


# --- Clean rules ---

cleanlib: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(FIND) $(BASE_OBJ_PATH) -name "*.o" | $(XARGS) $(RM_F)
	- $(FIND) $(BASE_LIB_PATH) -name "*.a" | $(XARGS) $(RM_F)
	- $(FIND) $(BASE_LIB_PATH) -name "*.so" | $(XARGS) $(RM_F)
else
	@echo "Removing .o files from $(BASE_OBJ_PATH)."
	@- $(FIND) $(BASE_OBJ_PATH) -name "*.o" | $(XARGS) $(RM_F)
	@echo "Removing .a files from $(BASE_LIB_PATH)."
	@- $(FIND) $(BASE_LIB_PATH) -name "*.a" | $(XARGS) $(RM_F)
	@echo "Removing .so files from $(BASE_LIB_PATH)."
	@- $(FIND) $(BASE_LIB_PATH) -name "*.so" | $(XARGS) $(RM_F)
endif

cleanh: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(RM_F) $(BLIS_H_FLAT)
else
	@echo "Removing blis.h file from $(BASE_INC_PATH)."
	@$(RM_F) $(BLIS_H_FLAT)
endif

cleantest: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(FIND) $(BASE_OBJ_TESTSUITE_PATH) \( -name "*.o" -o -name "*.pexe" \) | $(XARGS) $(RM_F)
	- $(RM_RF) $(TESTSUITE_BIN)
else
	@echo "Removing object files from $(BASE_OBJ_TESTSUITE_PATH)."
	@- $(FIND) $(BASE_OBJ_TESTSUITE_PATH) \( -name "*.o" -o -name "*.pexe" \) | $(XARGS) $(RM_F)
	@echo "Removing $(TESTSUITE_BIN) binary."
	@- $(RM_RF) $(TESTSUITE_BIN)
endif

cleanmk: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(FIND) $(CONFIG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(FRAME_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(REFKERN_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(KERNELS_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
else
	@echo "Removing makefile fragments from $(CONFIG_PATH)."
	@- $(FIND) $(CONFIG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(FRAME_PATH)."
	@- $(FIND) $(FRAME_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(REFKERN_PATH)."
	@- $(FIND) $(REFERKN_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(KERNELS_PATH)."
	@- $(FIND) $(KERNELS_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
endif

distclean: check-env cleanmk cleanlib cleanh cleantest
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $(CONFIG_MK_FILE)
	- $(RM_RF) $(TESTSUITE_OUT_FILE)
	- $(RM_RF) $(OBJ_DIR)
	- $(RM_RF) $(LIB_DIR)
	- $(RM_RF) $(INCLUDE_DIR)
else
	@echo "Removing $(CONFIG_MK_FILE)."
	@- $(RM_F) $(CONFIG_MK_FILE)
	@echo "Removing $(TESTSUITE_OUT_FILE)."
	@- $(RM_F) $(TESTSUITE_OUT_FILE)
	@echo "Removing $(OBJ_DIR)."
	@- $(RM_RF) $(OBJ_DIR)
	@echo "Removing $(LIB_DIR)."
	@- $(RM_RF) $(LIB_DIR)
	@echo "Removing $(INCLUDE_DIR)."
	@- $(RM_RF) $(INCLUDE_DIR)
endif


# --- CHANGELOG rules ---

changelog: check-env
	@echo "Updating '$(DIST_PATH)/$(CHANGELOG)' via '$(GIT_LOG)'."
	@$(GIT_LOG) > $(DIST_PATH)/$(CHANGELOG) 


# --- Uninstall rules ---

# NOTE: We can't write these uninstall rules directly in terms of targets
# $(MK_LIBS_INST_W_VERS_CONF), $(MK_LIBS_INST), and $(MK_INCL_DIR_INST)
# because those targets are already defined in terms of rules that *build*
# those products.

uninstall-libs: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $(MK_LIBS_INST_W_VERS_CONF)
else
	@echo "Removing $(MK_LIBS_INST_W_VERS_CONF)."
	@- $(RM_F) $(MK_LIBS_INST_W_VERS_CONF)
endif

uninstall-lib-symlinks: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $(MK_LIBS_INST)
else
	@echo "Removing $(MK_LIBS_INST)."
	@- $(RM_F) $(MK_LIBS_INST)
endif

uninstall-headers: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_RF) $(MK_INCL_DIR_INST)
else
	@echo "Removing $(MK_INCL_DIR_INST)/."
	@- $(RM_RF) $(MK_INCL_DIR_INST)
endif

# --- Uninstall old rules ---

uninstall-old-libs: $(UNINSTALL_LIBS) check-env

uninstall-old-headers: $(UNINSTALL_HEADERS) check-env

$(UNINSTALL_LIBS): check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $@
else
	@echo "Removing $(@F) from $(@D)/."
	@- $(RM_F) $@
endif

$(UNINSTALL_HEADERS): check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $@
else
	@echo "Removing $(@F) from $(@D)/."
	@- $(RM_F) $@
endif

