if(CMAKE_EXPORT_COMPILE_COMMANDS)
	add_custom_target(compile_commands_copy ALL
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${CMAKE_BINARY_DIR}/compile_commands.json"
			"${CMAKE_SOURCE_DIR}/compile_commands.json"
		BYPRODUCTS "${CMAKE_SOURCE_DIR}/compile_commands.json"
		COMMENT "Copying IDE object ${CMAKE_BINARY_DIR}/compile_commands.json"
	)
endif()
